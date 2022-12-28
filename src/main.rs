#![no_std]
#![no_main]

#![feature(abi_efiapi)]
#![feature(maybe_uninit_slice)]
#![feature(alloc_error_handler)]
#![feature(allocator_api)]
#![feature(array_methods)]
#![feature(nonnull_slice_from_raw_parts)]
#![feature(pointer_byte_offsets)]
#![feature(int_roundings)]

extern crate alloc;
extern crate core;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::{mem, ptr};
use core::cell::UnsafeCell;
use core::ffi::CStr;
use core::fmt::Write;
use core::iter::FromIterator;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicPtr, Ordering::*};

use elf_rs::{Elf, ProgramType};
use num_integer::Integer;
use uefi::{CStr16, Guid, Handle};
use uefi::prelude::cstr16;
use uefi::proto::device_path::{DevicePath, DevicePathNodeEnum, DeviceSubType, DeviceType};
use uefi::proto::device_path::media::PartitionSignature;
use uefi::proto::media::file::{Directory, File, FileAttribute, FileMode, FileType, RegularFile};
use uefi::proto::media::fs::SimpleFileSystem;
use uefi::table::boot::{OpenProtocolAttributes, OpenProtocolParams, ScopedProtocol, SearchType};

use prebootlib::KernelEntryFn;

use crate::boot_alloc::GlobalAllocBootUefi;
use crate::fixed_buf::FixedBufferUefi;

pub mod fixed_buf;
pub mod boot_alloc;

mod uefip {
	pub use uefi::data_types::Char16;
	pub use uefi::prelude::*;
	pub use uefi::proto::console::text::{Color, Output};
}

#[global_allocator]
static GLOBAL_ALLOC_BOOT: GlobalAllocBootUefi = GlobalAllocBootUefi::new_uninit();

static STATIC_BOOT_SERVICES_PTR: AtomicPtr<uefip::BootServices> = AtomicPtr::new(ptr::null_mut());

mod global_print {
	use core::mem;
	use core::sync::atomic::Ordering::*;
	
	use atomic::Atomic;
	use uefi::proto::console::text::Output;
	
	#[deprecated(note = "Internal only, don't use!")]
	pub static GLOBAL_UEFI_STDOUT_PTR: Atomic<usize> = Atomic::new(0x0);
	
	pub unsafe fn load_stdout(stdout: &mut Output) {
		#[allow(deprecated)]
		GLOBAL_UEFI_STDOUT_PTR.store(mem::transmute(stdout), SeqCst);
	}
	
	pub unsafe fn global_stdout<'a>() -> Option<&'a mut Output<'a>> {
//		let line = ::core::line!();
		
		// SAFETY: This looks massively unsafe because we are turning a pointer into a &mut
		//  without any sort of synchronization and really, it _is_.
		//  The uefi-rs lib also freely produces aliasing &mut Output though.
		//  This should be safe (TM) as the uefi-rs Output struct is only a repr(C) "window"
		//  to the immutable UEFI EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL and the mut-ness is purely
		//  needed for core::fmt::Write to work (as it needs a &mut receiver).
		//  So long story short, this is _still_ super UB (I think?) but should work robustly.
		#[allow(deprecated)]
		let stdout_ptr_usize = crate::global_print::GLOBAL_UEFI_STDOUT_PTR.load(SeqCst);
		(mem::transmute::<_, *mut Output<'a>>(stdout_ptr_usize)).as_mut()
	}
	
	/// Boot println! that prints to the uefi standard out.
	/// Only usable before boot services have been exited!
	#[macro_export]
	macro_rules! btprintln {
		() => {
			{
				if let Some(stdout) = unsafe {crate::global_print::global_stdout()} {
					let _ = ::core::writeln!(stdout);
				}
			}
		};
		($fmt_str:literal $(, $($args:expr),*)?) => {
			{
				if let Some(stdout) = unsafe {crate::global_print::global_stdout()} {
					let _ = ::core::writeln!(stdout, $fmt_str, $($($args),*)?);
				}
			}
		};
	}
}

// Don't use uefi::entry attrib because it assumes that uefi_rs is called uefi (which it isn't because we renamed it in Cargo.toml)
//#[entry]
#[no_mangle]
pub extern "efiapi" fn efi_main(img_handle: uefip::Handle, mut sys_table: uefip::SystemTable<uefip::Boot>) -> uefip::Status {
	// Store static boot services ptr
	// (This should ideally be done as early as possible
	//  so the panic_handler has a valid pointer, should smth. panic)
	STATIC_BOOT_SERVICES_PTR.store(sys_table.boot_services() as *const _ as *mut _, SeqCst);
	
	// Init global alloc
	unsafe {
		GLOBAL_ALLOC_BOOT.init_boot_alloc(sys_table.boot_services());
	}
	
	// Get uefi stdout handle
	unsafe {
		global_print::load_stdout(sys_table.stdout());
	}
	
	// Log hello world
//	stdout.set_color(uefip::Color::White, uefip::Color::Blue).unwrap().unwrap();
	btprintln!("Hello World (from bootloader_uefi!)");
	btprintln!();
	
	// Find boot part handle
	let mut boot_part_sfs_prot = Option::<ScopedProtocol<SimpleFileSystem>>::None;
	{
		let sfs_handles = {
			let buf_len = sys_table.boot_services()
				.locate_handle(SearchType::from_proto::<SimpleFileSystem>(), None)
				.unwrap();
			
			let mut handle_buf: Vec<MaybeUninit<Handle>> = (0..buf_len)
				.map(|_| MaybeUninit::zeroed())
				.collect();
			
			sys_table.boot_services()
				.locate_handle(SearchType::from_proto::<SimpleFileSystem>(), Some(&mut handle_buf))
				.unwrap();
			
			handle_buf
		};
		
		for sfs_handle in sfs_handles.iter().map(|maybe_uninit_handle| unsafe {maybe_uninit_handle.assume_init()}) {
			let open_params = OpenProtocolParams {
				handle: sfs_handle,
				agent: img_handle,
				controller: None,
			};
			let proto_res = unsafe {
				sys_table.boot_services()
					.open_protocol::<DevicePath>(open_params, OpenProtocolAttributes::GetProtocol)
			};
			
			match proto_res {
				Ok(succ) => {
					btprintln!("handle {:?}", (unsafe {mem::transmute::<_, *const u8>(sfs_handle)}));
					
					for node in succ.node_iter() {
						btprintln!("  {:?} {:?}", node.device_type(), node.sub_type());
						
						if node.device_type() == DeviceType::MEDIA && node.sub_type() == DeviceSubType::MEDIA_HARD_DRIVE {
							let node_enum = node.as_enum().unwrap();
							let hd_node = match node_enum {
								DevicePathNodeEnum::MediaHardDrive(node) => node,
								_ => panic!(),
							};
							
							let boot_partn_guid_bytes = Guid::from_bytes(0xA4A4A4A4_A4A4_A4A4_A4A4_A4A4A4A4A4A4u128.to_le_bytes());
							
							match hd_node.partition_signature() {
								PartitionSignature::Guid(guid_sig) => {
									if guid_sig == boot_partn_guid_bytes {
										btprintln!("  found boot partn, handle is the boot partn handle");
										
										let sfs_proto = unsafe {
											sys_table.boot_services()
												.open_protocol(OpenProtocolParams {
													handle: sfs_handle,
													agent: img_handle,
													controller: None,
												}, OpenProtocolAttributes::GetProtocol)
										};
										boot_part_sfs_prot = Some(sfs_proto.unwrap());
									}
								}
								_ => panic!(),
							}
						}
					}
				}
				Err(_) => btprintln!("dev doesn't support device_handle protocol"),
			}
		}
	}
	
//	// DEBUG:
//	writeln!(stdout, "got handles:").unwrap();
//	for h in &sfs_handles {
//		writeln!(stdout, "handle {:?}", unsafe {mem::transmute_copy::<_, usize>(h)}).unwrap();	
//	}
	
	let boot_part_sfs_prot = boot_part_sfs_prot
		.expect("Failed to find boot partn handle");
	
	// Load kernel
	let kernel_file = {
		let mut sfs_prot: ScopedProtocol<SimpleFileSystem> = boot_part_sfs_prot;
		let mut directory = sfs_prot.open_volume()
			.unwrap();
		
		// DEBUG: Print entries
		let mut buf = Box::new([0u8; 512]);
		directory.reset_entry_readout()
			.unwrap();
		
		while let Some(entry_info) = directory.read_entry(buf.as_mut_slice()).unwrap() {
			btprintln!("file \"{}\"", entry_info.file_name());
		}
		
		// Load kernel.elf
		read_sfs_file(&mut directory, cstr16!("kernel.elf")).unwrap()
	};
	
	btprintln!("kernel.elf file size: {}", kernel_file.len());
	
	// Parse kernel elf
	let kernel_elf = Elf::from_bytes(&kernel_file).unwrap();
	let kernel_elf = if let Elf::Elf64(elf) = kernel_elf {
		elf
	} else {
		panic!("kernel.elf isn't a valid elf64");
	};
	
	btprintln!();
	
//	// DEBUG: Try jumping to the kernel
//	let text_section = kernel_elf.lookup_section(".text").unwrap();
	
//	btprintln!("elf entry point {:08x}", kernel_elf.header().entry_point());
//	btprintln!("actual entry point offset {:08x}", entry_offset);
	
//	btprintln!("entry point instructions:\n{:x?}", &kernel_file[(entry_offset as usize)..(entry_offset as usize + 32)]);
	
	// Load kernel image to fixed addr
	let static_elf_base = 0x2000000;
	
	const PAGE_SIZE: usize = 4096;
	let ph_max_addr = kernel_elf.program_headers().iter()
		.filter(|ph| ph.ph_type() == ProgramType::LOAD)
		.inspect(|ph| btprintln!("ph vaddr {}, memsz {}", ph.vaddr(), ph.memsz()))
		.fold(0, |prev, ph| usize::max(prev, (ph.vaddr() + ph.memsz()).saturating_sub(static_elf_base as _) as usize));
	
	let mut kernel_img = FixedBufferUefi::alloc_at(0x2_000_000, ph_max_addr.div_ceil(PAGE_SIZE)).unwrap();
	
	unsafe {
		// Copy segments to mem
		for ph in kernel_elf.program_headers() {
			if ph.ph_type() == ProgramType::LOAD && ph.vaddr() >= static_elf_base {
				let src_ptr = kernel_file.as_ptr().offset(ph.offset() as isize);
				let dst_ptr = kernel_img.as_mut_ptr().offset(ph.vaddr() as isize - static_elf_base as isize);
				
				// Copy data
				ptr::copy_nonoverlapping(src_ptr, dst_ptr, ph.filesz() as usize);
				
				// Zero out rest
				let rest_ptr = kernel_img.as_mut_ptr().offset(((ph.vaddr() - static_elf_base as u64) + ph.filesz()) as isize);
				ptr::write_bytes(rest_ptr, 0, ph.memsz().saturating_sub(ph.filesz()) as usize);
			}
		}
	}
	
	btprintln!("loaded kernel img at fixed addr {:08x} with len {}", kernel_img.start_addr(), kernel_img.as_slice().len());
	
	// DEBUG: Dump mem
//	btprintln!(".rodata dump: {:?}", unsafe {core::str::from_utf8_unchecked(&kernel_img.as_slice()[0x741C0..0x741D0])});
//	btprintln!(".rodata dump: {:?}", &kernel_img.as_slice()[0x74120..0x74140]);
	
//	for _ in 0..usize::MAX {}
	
	// Transfer control to the kernel
	let kernel_entry_addr = kernel_img.start_addr() + kernel_elf.header().entry_point() as usize - static_elf_base as usize;
	unsafe {
		mem::transmute::<_, KernelEntryFn>(kernel_entry_addr as *const u8)(img_handle, sys_table)
	}
	
//	btprintln!("Called kernel, got magic {}.{}",magic>>16, magic&0xFFFF);
	
//	let mut magic: u32;
//	unsafe {
//		asm!(
//			"jmp {0}"
//			in(reg) entrypoint_addr,
//			out("ax") magic,
//		);
//	}
	
//	// Return status to firmware
//	uefi_rs::Status::SUCCESS
}

fn read_sfs_file(root: &mut Directory, path: &CStr16) -> Result<Vec<u8>, ()> {
	let file_handle = root.open(path, FileMode::Read, FileAttribute::from_bits(0).unwrap())
		.expect("Failed to open file");
	
	if let FileType::Regular(mut file) = file_handle.into_type().unwrap() {
		// Query size and alloc buffer
//		let (_, size) = file.read(&mut []).unwrap().split();
		file.set_position(RegularFile::END_OF_FILE).unwrap();
		let size = file.get_position().unwrap() as usize;
		
//		let buf_ptr = alloc::alloc::Global::default().allocate(core::alloc::Layout::from_size_align(size, 1).unwrap()).unwrap();
//		let mut data = Vec::from(unsafe {Box::from_raw(buf_ptr.as_ptr())});
//		let mut data = Vec::with_capacity(size);
//		unsafe {data.set_len(size)};
//		let mut data = Vec::from_iter(core::iter::repeat(0u8).take(size));
		let mut data = alloc::vec![0u8; size]; // vec! is currently the fastest way of allocating a zeroed vec
		
//		unsafe {
//			let test_mem = alloc::alloc::alloc(core::alloc::Layout::from_size_align(size, 1).unwrap());
//			let is_zeroed = core::slice::from_raw_parts(test_mem, size).iter().copied()
//				.all(|b| b == 0);
//			
//			kprintln!("uefi alloc'ed zeroed mem? {}", is_zeroed);
//		}
		
		// Read file
		file.set_position(0).unwrap();
		file.read(&mut data).unwrap();
		
		Ok(data)
	} else {
		Err(())
	}
}

#[panic_handler]
fn panic_handler(info: &core::panic::PanicInfo) -> ! {
	// Try load static boot_services pointer
	let maybe_boot_srvc = unsafe {
		STATIC_BOOT_SERVICES_PTR.load(SeqCst)
			.as_ref()
	};
	
	// DEBUG: Print panic info
//	let stdout = unsafe {(TEST_STDOUT_PTR.load(SeqCst) as *mut uefip::Output).as_mut()};
	let stdout = unsafe {global_print::global_stdout()};
	
	if let Some(stdout) = stdout {
		let _ = stdout.set_color(uefip::Color::LightRed, uefip::Color::Black);
		let _ = write!(stdout, "{}", info);
	} else {
		// Well, shit...
	}
	
	if let Some(boot_srvc) = maybe_boot_srvc {
		type FnEfiExit = extern "efiapi" fn(image_handle: uefip::Handle, exit_status: uefip::Status, exit_data_size: usize, exit_data: *const uefip::Char16);
		
		let exit_fn_offset = 
			mem::size_of::<uefi::table::Header>()
			+ 24 * mem::size_of::<usize>();
		let _exit_fn_ptr: FnEfiExit = unsafe {
			mem::transmute((boot_srvc as *const _ as *const u8)
				.offset(exit_fn_offset as isize))
		};
		
		// exit_fn_ptr()
		
		loop {}
	}
	else {
		// We didn't have a valid boot_services pointer (that's a bug!)
		// so just loop forever and chill
		loop {}
	}
}

#[alloc_error_handler]
fn alloc_error_handler(layout: core::alloc::Layout) -> ! {
    panic!("Memory allocation failed: {:?}", layout)
}

pub trait AsPtr {
	fn as_ptr(&self) -> *const Self;
}
pub trait AsPtrMut {
	fn as_ptr_mut(&mut self) -> *mut Self;
}
impl<T> AsPtr for T {
	fn as_ptr(&self) -> *const Self {
		self as *const Self
	}
}
impl<T> AsPtrMut for T {
	fn as_ptr_mut(&mut self) -> *mut Self {
		self as *mut Self
	}
}

pub trait PtrOpsExt {
	unsafe fn byte_offset(self, offset: isize) -> Self;
}
impl<T> PtrOpsExt for *const T {
	unsafe fn byte_offset(self, offset: isize) -> Self {
		self.cast::<u8>()
			.offset(offset)
			.cast()
	}
}
impl<T> PtrOpsExt for *mut T {
	unsafe fn byte_offset(self, offset: isize) -> Self {
		self.cast::<u8>()
			.offset(offset)
			.cast()
	}
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub enum RawDeviceType {
	Hardware = 0x01,
	ACPI = 0x02,
	Messaging = 0x03,
	Media = 0x04,
	BIOSBootSpec = 0x05,
	End = 0x7F,
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub enum RawDeviceSubTypeEnd {
	EndInstance = 0x01,
	EndEntire = 0xFF,
}

//pub struct DevPathChain {
//	path_ptr: *const DevPath,
//}
//
//pub struct DevPathIter {
//	
//}
//impl Iterator for DevPathIter {
//	type Item = RawDevPathNode;
//	
//	fn next(&mut self) -> Option<Self::Item> {
//		unimplemented!()
//	}
//}
//
//pub struct RawDevPathNode {
//	
//}
