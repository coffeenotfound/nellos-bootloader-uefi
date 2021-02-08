use core::{mem, slice};
use core::ptr::NonNull;
use core::sync::atomic::Ordering::SeqCst;

use fallo::alloc::AllocError;
use uefi_rs::table::boot::MemoryType;

use crate::boot_alloc::{AllocPagesFn, BOOT_ALLOC_PAGES_FN};
use crate::uefip::Status;

pub struct FixedBufferUefi {
	start_addr: usize,
	ptr: NonNull<u8>,
	num_pages: usize,
}

#[repr(u32)]
enum EfiAllocType {
	AnyPages = 0,
	MaxAddres = 1,
	Address = 2,
}

impl FixedBufferUefi {
	pub const PAGE_SIZE: usize = 4096;
	
	pub fn alloc_at(start_addr: usize, num_pages: usize) -> Result<FixedBufferUefi, Status> {
		if let Some(f) = NonNull::new(BOOT_ALLOC_PAGES_FN.load(SeqCst) as *mut ()) {
			unsafe {
				let mut addr: u64 = start_addr as u64;
				match mem::transmute::<_, AllocPagesFn>(f.as_ptr())(EfiAllocType::Address as u32, MemoryType::LOADER_DATA, num_pages, &mut addr) {
					s if s.is_success() => {
							Ok(Self {
							start_addr,
							ptr: NonNull::new_unchecked(addr as usize as *mut u8),
							num_pages
						})
					}
					e => Err(e)
				}
			}
		} else {
			Err(Status::NOT_READY)
		}
	}
	
	pub fn start_addr(&self) -> usize {
		self.start_addr
	}
	
	pub fn num_pages(&self) -> usize {
		self.num_pages
	}
	
	pub fn byte_size(&self) -> usize {
		self.num_pages * Self::PAGE_SIZE
	}
	
	pub fn as_mut_ptr(&mut self) -> *mut u8 {
		self.ptr.as_ptr()
	}
	
	pub fn as_mut_slice(&mut self) -> &mut [u8] {
		unsafe {slice::from_raw_parts_mut(self.ptr.as_ptr(), self.num_pages * Self::PAGE_SIZE)}
	}
	
	pub fn as_slice(&self) -> &[u8] {
		unsafe {slice::from_raw_parts(self.ptr.as_ptr(), self.num_pages * Self::PAGE_SIZE)}
	}
}
