#![no_std]
#![no_main]

#![feature(abi_efiapi)]
#![feature(maybe_uninit_slice)]
#![feature(alloc_error_handler)]
#![feature(allocator_api)]
#![feature(const_raw_ptr_to_usize_cast)]
#![feature(array_methods)]
#![feature(nonnull_slice_from_raw_parts)]

extern crate alloc;
extern crate core;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::{mem, ptr};
use core::cell::UnsafeCell;
use core::fmt::Write;
use core::iter::FromIterator;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicPtr, Ordering::*};

use elf_rs::Elf;
use num_integer::Integer;
use uefi_rs::Handle;
use uefi_rs::proto::loaded_image::DevicePath;
use uefi_rs::proto::media::file::{Directory, File, FileAttribute, FileMode, FileType, RegularFile};
use uefi_rs::proto::media::fs::SimpleFileSystem;
use uefi_rs::table::boot::SearchType;

use prebootlib::KernelEntryFn;

use crate::boot_alloc::GlobalAllocBootUefi;
use crate::fixed_buf::FixedBufferUefi;

pub mod fixed_buf;
pub mod boot_alloc;

mod uefip {
	pub use uefi_rs::data_types::Char16;
	pub use uefi_rs::prelude::*;
	pub use uefi_rs::proto::console::text::{Color, Output};
}

#[global_allocator]
static GLOBAL_ALLOC_BOOT: GlobalAllocBootUefi = GlobalAllocBootUefi::new_uninit();

static STATIC_BOOT_SERVICES_PTR: AtomicPtr<uefip::BootServices> = AtomicPtr::new(ptr::null_mut());

mod global_print {
	use core::mem;
	use core::sync::atomic::Ordering::*;
	
	use atomic::Atomic;
	use uefi_rs::proto::console::text::Output;
	
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
pub extern "efiapi" fn efi_main(img_handle: uefip::Handle, sys_table: uefip::SystemTable<uefip::Boot>) -> uefip::Status {
	// Store static boot services ptr
	// (This should ideally be done as early as possible
	//  so the panic_handler has a valid pointer, should smth. panic)
	STATIC_BOOT_SERVICES_PTR.store(sys_table.boot_services() as *const _ as *mut _, SeqCst);
	
	// Init global alloc
	unsafe {
		GLOBAL_ALLOC_BOOT.init_boot_alloc(sys_table.boot_services());
	}
	
	// Get uefi stdout handle
	let stdout = sys_table.stdout();
	unsafe {
		global_print::load_stdout(stdout);
	}
	
	// Log hello world
//	stdout.set_color(uefip::Color::White, uefip::Color::Blue).unwrap().unwrap();
	btprintln!("Hello World (from bootloader_uefi!)");
	btprintln!();
	
	// Find boot part handle
	let mut boot_part_sfs_prot = Option::<&UnsafeCell<SimpleFileSystem>>::None;
	{
		let sfs_handles = {
			let buf_len = sys_table.boot_services()
				.locate_handle(SearchType::from_proto::<SimpleFileSystem>(), None)
				.unwrap().unwrap();
			
			let mut handle_buf = Vec::from_iter((0..buf_len).map(|_| MaybeUninit::<Handle>::zeroed()));
			let handle_buf_slice_init = unsafe {
				MaybeUninit::slice_assume_init_mut(handle_buf.as_mut_slice())
			};
			sys_table.boot_services()
				.locate_handle(SearchType::from_proto::<SimpleFileSystem>(), Some(handle_buf_slice_init))
				.unwrap().unwrap();
			
//			let handle_buf = unsafe {
//				mem::transmute::<_, Vec<Handle>>(handle_buf)
//			};
			unsafe {
				mem::transmute::<_, Vec<Handle>>(handle_buf)
			}
		};
		
		for &handle in &sfs_handles {
			match sys_table.boot_services().handle_protocol::<DevicePath>(handle) {
				Ok(succ) => {
					let (_status, dp_prot) = succ.split();
					let dev_path = unsafe {&*dp_prot.get()};
					
					btprintln!("handle {:?}", (unsafe {mem::transmute::<_, *const u8>(handle)}));
					
					let mut node_ref = dev_path;
					loop {
						let (dev_type, subtype) = unsafe {
							(mem::transmute_copy::<_, RawDeviceType>(&node_ref.device_type),
							mem::transmute_copy::<_, RawDeviceSubTypeEnd>(&node_ref.sub_type))
						};
						let node_len = u16::from_le_bytes(node_ref.length);
						
						btprintln!("  {:?} {} len={}", node_ref.device_type, unsafe {mem::transmute::<_, u8>(subtype)}, node_len);
						
						if let RawDeviceType::Media = dev_type {
							if subtype as u8 == 1 {
								let sig_type;
								let partn_sig;
								unsafe {
									sig_type = *node_ref.as_ptr().byte_offset(41).cast::<u8>();
									partn_sig = *node_ref.as_ptr().byte_offset(24).cast::<[u8; 16]>();
								}
								
								let boot_partn_guid_bytes = 0xA4A4A4A4_A4A4_A4A4_A4A4_A4A4A4A4A4A4u128.to_le_bytes();
								
								const SIG_TYPE_GUID_PARTN: u8 = 0x02;
								if sig_type == SIG_TYPE_GUID_PARTN && partn_sig == boot_partn_guid_bytes {
									boot_part_sfs_prot = Some(
										sys_table.boot_services()
											.handle_protocol::<SimpleFileSystem>(handle)
											.unwrap().unwrap()
									);
									
									btprintln!("  found boot partn, handle is the boot part handle");
								}
							}
						}
						
						if let RawDeviceType::End = dev_type {
							break;
						} else {
							node_ref = unsafe {
								&*node_ref.as_ptr().byte_offset(node_len as isize)
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
		let sfs_prot = unsafe {&mut *boot_part_sfs_prot.get()};
		let mut directory = sfs_prot.open_volume().unwrap().unwrap();
		
		// DEBUG: Print entries
		let mut buf = Box::new([0u8; 512]);
		directory.reset_entry_readout().unwrap().unwrap();
		
		while let Some(entry_info) = directory.read_entry(buf.as_mut_slice()).unwrap().unwrap() {
			btprintln!("file \"{}\"", entry_info.file_name());
		}
		
		// Load kernel.elf
		read_sfs_file(&mut directory, "kernel.elf").unwrap()
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
	
	// DEBUG:
	write!(stdout, "{}\n", 2);
	
//	// DEBUG: Try jumping to the kernel
//	let text_section = kernel_elf.lookup_section(".text").unwrap();
	
//	btprintln!("elf entry point {:08x}", kernel_elf.header().entry_point());
//	btprintln!("actual entry point offset {:08x}", entry_offset);
	
//	btprintln!("entry point instructions:\n{:x?}", &kernel_file[(entry_offset as usize)..(entry_offset as usize + 32)]);
	
	// Load kernel image to fixed addr
	const PAGE_SIZE: usize = 4096;
	let ph_max_addr = kernel_elf.program_headers().iter()
		.fold(0, |prev, ph| usize::max(prev, (ph.vaddr() + ph.memsz()) as usize));
	
	let mut kernel_img = FixedBufferUefi::alloc_at(0x2_000_000, ph_max_addr.div_ceil(&PAGE_SIZE)).unwrap();
	
	unsafe {
		// Zero out memory
		ptr::write_bytes(kernel_img.as_mut_slice().as_mut_ptr(), 0, kernel_img.byte_size());
		
		// Copy segments to mem
		for ph in kernel_elf.program_headers() {
			let src_ptr = kernel_file.as_ptr().offset(ph.offset() as isize);
			let dst_ptr = kernel_img.as_mut_ptr().offset(ph.vaddr() as isize);
			
			ptr::copy_nonoverlapping(src_ptr, dst_ptr, ph.filesz() as usize);
		}
	}
	
	btprintln!("loaded kernel img at fixed addr {:08x}", kernel_img.start_addr());
	
	// Transfer control to the kernel
	let kernel_entry_addr = kernel_img.start_addr() + kernel_elf.header().entry_point() as usize;
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

fn read_sfs_file(root: &mut Directory, path: &str) -> Result<Vec<u8>, ()> {
	let file_handle = root.open(path, FileMode::Read, FileAttribute::from_bits(0).unwrap())
		.expect("Failed to open file").unwrap();
	
	if let FileType::Regular(mut file) = file_handle.into_type().unwrap().unwrap() {
		// Query size and alloc buffer
//		let (_, size) = file.read(&mut []).unwrap().split();
		file.set_position(RegularFile::END_OF_FILE).unwrap().unwrap();
		let size = file.get_position().unwrap().unwrap() as usize;
		
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
		file.set_position(0).unwrap().unwrap();
		match file.read(&mut data).map_err(|_| ())?.split() {
			(s, _) if s.is_success() => Ok(()),
			(_, _) => Err(()),
		}?;
		
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
			mem::size_of::<uefi_rs::table::Header>()
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
