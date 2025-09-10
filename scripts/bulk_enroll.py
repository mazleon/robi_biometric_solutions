#!/usr/bin/env python3
"""
Bulk Face Enrollment Script

This script processes all images in a specified folder and enrolls them
using the face verification API. It includes comprehensive logging and
monitoring capabilities.

Usage:
    python scripts/bulk_enroll.py --folder "D:/RnD/face-rnd/UTK_Face_Data"
"""

import os
import sys
import asyncio
import aiohttp
import aiofiles
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from datetime import datetime
import json
import traceback

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import settings

# Configure logging with UTF-8 encoding
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bulk_enrollment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BulkEnrollmentStats:
    """Track enrollment statistics and progress."""
    
    def __init__(self):
        self.total_files = 0
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = None
        self.end_time = None
        self.errors = []
        self.successful_enrollments = []
        
    def start(self):
        """Mark the start of processing."""
        self.start_time = time.time()
        logger.info(f"Starting bulk enrollment of {self.total_files} files")
        
    def finish(self):
        """Mark the end of processing."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"Bulk enrollment completed in {duration:.2f} seconds")
        self.print_summary()
        
    def record_success(self, user_id: str, file_path: str, response_data: dict):
        """Record a successful enrollment."""
        self.successful += 1
        self.processed += 1
        self.successful_enrollments.append({
            'user_id': user_id,
            'file_path': file_path,
            'timestamp': datetime.now().isoformat(),
            'response': response_data
        })
        logger.info(f"Successfully enrolled {user_id} from {Path(file_path).name}")
        
    def record_failure(self, user_id: str, file_path: str, error: str):
        """Record a failed enrollment."""
        self.failed += 1
        self.processed += 1
        self.errors.append({
            'user_id': user_id,
            'file_path': file_path,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        logger.error(f"Failed to enroll {user_id} from {Path(file_path).name}: {error}")
        
    def record_skip(self, file_path: str, reason: str):
        """Record a skipped file."""
        self.skipped += 1
        self.processed += 1
        logger.warning(f"Skipped {Path(file_path).name}: {reason}")
        
    def print_progress(self):
        """Print current progress."""
        if self.total_files > 0:
            progress = (self.processed / self.total_files) * 100
            logger.info(f"Progress: {self.processed}/{self.total_files} ({progress:.1f}%) - "
                       f"Success: {self.successful}, Failed: {self.failed}, Skipped: {self.skipped}")
            
    def print_summary(self):
        """Print final summary."""
        duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        logger.info("=" * 60)
        logger.info("BULK ENROLLMENT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files found: {self.total_files}")
        logger.info(f"Files processed: {self.processed}")
        logger.info(f"Successful enrollments: {self.successful}")
        logger.info(f"Failed enrollments: {self.failed}")
        logger.info(f"Skipped files: {self.skipped}")
        logger.info(f"Success rate: {(self.successful/max(1, self.processed))*100:.1f}%")
        logger.info(f"Total duration: {duration:.2f} seconds")
        logger.info(f"Average time per file: {duration/max(1, self.processed):.2f} seconds")
        logger.info("=" * 60)
        
        # Save detailed results to file
        self.save_results()
        
    def save_results(self):
        """Save detailed results to JSON file."""
        results = {
            'summary': {
                'total_files': self.total_files,
                'processed': self.processed,
                'successful': self.successful,
                'failed': self.failed,
                'skipped': self.skipped,
                'success_rate': (self.successful/max(1, self.processed))*100,
                'duration': self.end_time - self.start_time if self.end_time and self.start_time else 0,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                'end_time': datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
            },
            'successful_enrollments': self.successful_enrollments,
            'errors': self.errors
        }
        
        results_file = f"logs/bulk_enrollment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Detailed results saved to: {results_file}")


class BulkEnroller:
    """Handle bulk enrollment operations."""
    
    def __init__(self, api_base_url: str = "http://127.0.0.1:8000", id_source: str = "filename"):
        self.api_base_url = api_base_url.rstrip('/')
        self.session = None
        self.stats = BulkEnrollmentStats()
        # How to derive user_id from files: 'filename', 'parent', or 'path'
        self.id_source = id_source
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    def get_supported_extensions(self) -> set:
        """Get supported image file extensions (from settings with common variations)."""
        exts = set(ext.lower() for ext in getattr(settings, 'allowed_extensions', ['.jpg', '.jpeg', '.png', '.webp']))
        # Add common variants if not present
        if '.tiff' in exts or '.tif' in exts:
            exts.update({'.tif', '.tiff'})
        exts.update({'.bmp'})
        return exts
        
    def extract_user_id(self, file_path: Path, root_folder: Optional[Path] = None) -> str:
        """
        Extract user ID from file path according to the configured source.
        
        Options:
        - 'filename': use the filename stem (default; backward compatible)
        - 'parent'  : use the immediate parent folder name (good for datasets like VGG-Face)
        - 'path'    : use the relative path (without extension) from the root folder with separators replaced by '_'
        """
        try:
            if self.id_source == 'parent':
                return file_path.parent.name
            elif self.id_source == 'path' and root_folder is not None:
                rel = file_path.relative_to(root_folder)
                rel_no_ext = rel.with_suffix('')
                return str(rel_no_ext).replace('\\', '_').replace('/', '_')
            else:  # filename
                return file_path.stem
        except Exception:
            # Fallback to filename stem on any error
            return file_path.stem
        
    def find_image_files(self, folder_path: str) -> List[Path]:
        """Find all supported image files in the folder."""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
            
        supported_extensions = self.get_supported_extensions()
        image_files = []
        
        logger.info(f"Scanning folder: {folder_path}")
        
        for file_path in folder.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_files.append(file_path)
                
        logger.info(f"Found {len(image_files)} image files")
        return sorted(image_files)
        
    async def enroll_single_image(self, file_path: Path, user_id: str, name: Optional[str] = None, retries: int = 3) -> Tuple[bool, dict]:
        """
        Enroll a single image via the API.
        
        Returns:
            Tuple of (success: bool, response_data: dict)
        """
        try:
            # Prepare form data
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
                
            form_data = aiohttp.FormData()
            form_data.add_field('user_id', user_id)
            if name:
                form_data.add_field('name', name)
            # Provide metadata as JSON string so API can parse
            form_data.add_field('metadata', json.dumps({
                'source': 'bulk_enroll',
                'filename': file_path.name,
                'path': str(file_path),
                'id_source': self.id_source,
                'timestamp': datetime.now().isoformat()
            }))
            # Determine content type based on file extension
            ext = file_path.suffix.lower()
            content_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.bmp': 'image/bmp',
                '.tiff': 'image/tiff',
                '.tif': 'image/tiff',
                '.webp': 'image/webp'
            }
            ct = content_type_map.get(ext, 'application/octet-stream')
            form_data.add_field('file', file_content, 
                              filename=file_path.name, 
                              content_type=ct)
            
            # Make API request with retry/backoff
            attempt = 0
            last_error: Dict[str, any] = {}
            while attempt <= retries:
                try:
                    async with self.session.post(
                        f"{self.api_base_url}/api/v1/enroll",
                        data=form_data,
                        timeout=aiohttp.ClientTimeout(total=settings.request_timeout or 60)
                    ) as response:
                        try:
                            response_data = await response.json()
                        except Exception:
                            try:
                                text_response = await response.text()
                                response_data = {'error': f'Invalid JSON response: {text_response[:200]}...'}
                            except Exception:
                                response_data = {'error': f'Failed to parse response'}

                        if response.status == 200 and response_data.get('success', False):
                            return True, response_data

                        # Retry on 429 or 5xx
                        if response.status in (429, 500, 502, 503, 504) and attempt < retries:
                            wait = min(2 ** attempt, 10)
                            await asyncio.sleep(wait)
                            attempt += 1
                            continue

                        error_msg = (
                            response_data.get('detail') or 
                            response_data.get('message') or 
                            response_data.get('error') or 
                            f'HTTP {response.status}'
                        )
                        return False, {'error': error_msg, 'status': response.status, 'response': response_data}
                except asyncio.TimeoutError:
                    if attempt < retries:
                        wait = min(2 ** attempt, 10)
                        await asyncio.sleep(wait)
                        attempt += 1
                        last_error = {'error': 'Request timeout'}
                        continue
                    return False, {'error': 'Request timeout'}
                except aiohttp.ClientError as e:
                    if attempt < retries:
                        wait = min(2 ** attempt, 10)
                        await asyncio.sleep(wait)
                        attempt += 1
                        last_error = {'error': f'Network error: {str(e)}'}
                        continue
                    return False, {'error': f'Network error: {str(e)}'}
            return False, last_error or {'error': 'Unknown error after retries'}
                    
        except asyncio.TimeoutError:
            return False, {'error': 'Request timeout'}
        except aiohttp.ClientError as e:
            return False, {'error': f'Network error: {str(e)}'}
        except Exception as e:
            return False, {'error': f'Unexpected error: {str(e)}'}
            
    async def check_api_health(self) -> bool:
        """Check if the API is healthy and accessible."""
        try:
            async with self.session.get(
                f"{self.api_base_url}/api/v1/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"API is healthy: {data.get('status', 'unknown')}")
                    return True
                else:
                    logger.error(f"API health check failed: HTTP {response.status}")
                    return False
        except Exception as e:
            logger.error(f"API health check failed: {str(e)}")
            return False
            
    async def process_folder(self, folder_path: str, max_concurrent: int = 5, 
                           dry_run: bool = False,
                           retries: int = 3,
                           resume: bool = False,
                           checkpoint_path: Optional[str] = None) -> BulkEnrollmentStats:
        """
        Process all images in the folder.
        
        Args:
            folder_path: Path to the folder containing images
            max_concurrent: Maximum number of concurrent API requests
            dry_run: If True, only scan files without enrolling
        """
        # Check API health first
        if not dry_run and not await self.check_api_health():
            raise RuntimeError("API is not accessible. Please ensure the server is running.")
            
        # Find all image files
        image_files = self.find_image_files(folder_path)
        self.stats.total_files = len(image_files)
        
        if self.stats.total_files == 0:
            logger.warning("No image files found in the specified folder")
            return self.stats
            
        if dry_run:
            logger.info(f"DRY RUN: Would process {self.stats.total_files} files")
            root = Path(folder_path)
            for file_path in image_files[:10]:  # Show first 10 as example
                user_id = self.extract_user_id(file_path, root)
                logger.info(f"  - {file_path.name} -> user_id: {user_id}")
            if len(image_files) > 10:
                logger.info(f"  ... and {len(image_files) - 10} more files")
            return self.stats
            
        self.stats.start()
        
        # Load checkpoint if resume is enabled
        processed_set = set()
        ckpt_file = checkpoint_path or 'logs/bulk_enrollment_checkpoint.json'
        if resume and os.path.exists(ckpt_file):
            try:
                with open(ckpt_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    processed_set = set(data.get('processed_files', []))
                logger.info(f"Resume enabled: loaded {len(processed_set)} processed entries from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")

        # Process files with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        root = Path(folder_path)

        async def process_single_file(file_path: Path):
            async with semaphore:
                try:
                    # Skip if already processed in resume mode
                    if resume and str(file_path) in processed_set:
                        self.stats.record_skip(str(file_path), "Already processed (resume)")
                        return
                    # Check if file still exists and is readable
                    if not file_path.exists():
                        self.stats.record_skip(str(file_path), "File not found")
                        return
                        
                    if file_path.stat().st_size == 0:
                        self.stats.record_skip(str(file_path), "Empty file")
                        return
                        
                    # Respect application configured max file size
                    max_size = getattr(settings, 'max_file_size', 5 * 1024 * 1024)
                    if file_path.stat().st_size > max_size:
                        self.stats.record_skip(str(file_path), f"File too large (>{max_size} bytes)")
                        return
                    
                    user_id = self.extract_user_id(file_path, root)
                    
                    # Extract name from UTK dataset filename if possible
                    # Format: [age]_[gender]_[race]_[date&time].jpg
                    try:
                        parts = file_path.stem.split('_')
                        if len(parts) >= 3:
                            age, gender, race = parts[:3]
                            name = f"Person_{age}y_{gender}_{race}"
                        else:
                            name = f"Person_{user_id}"
                    except:
                        name = f"Person_{user_id}"
                    
                    success, response_data = await self.enroll_single_image(file_path, user_id, name, retries=retries)
                    
                    if success:
                        self.stats.record_success(user_id, str(file_path), response_data)
                        # Update checkpoint on success
                        if resume:
                            try:
                                processed_set.add(str(file_path))
                                os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
                                with open(ckpt_file, 'w', encoding='utf-8') as f:
                                    json.dump({'processed_files': list(processed_set)}, f)
                            except Exception as e:
                                logger.warning(f"Failed to update checkpoint: {e}")
                    else:
                        error_msg = response_data.get('error', 'Unknown error')
                        self.stats.record_failure(user_id, str(file_path), error_msg)
                        
                except Exception as e:
                    # Handle any unexpected errors during file processing
                    user_id = self.extract_user_id(file_path, root) if file_path.exists() else "unknown"
                    self.stats.record_failure(user_id, str(file_path), f"File processing error: {str(e)}")
                
                # Print progress every 10 files
                if self.stats.processed % 10 == 0:
                    self.stats.print_progress()
        
        # Process files concurrently without creating all tasks at once (memory-friendly)
        pending = set()
        iterator = iter(image_files)
        try:
            # Prime initial tasks
            for _ in range(min(max_concurrent * 4, len(image_files))):
                try:
                    fp = next(iterator)
                except StopIteration:
                    break
                pending.add(asyncio.create_task(process_single_file(fp)))

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                # Refill pending up to window size
                for _ in range(len(done)):
                    try:
                        fp = next(iterator)
                        pending.add(asyncio.create_task(process_single_file(fp)))
                    except StopIteration:
                        break
        finally:
            # Await any remaining tasks
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        
        self.stats.finish()
        return self.stats


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Bulk Face Enrollment Script')
    parser.add_argument('--folder', '-f', required=True,
                       help='Path to folder containing images')
    parser.add_argument('--api-url', default='http://127.0.0.1:8000',
                       help='Base URL of the face verification API')
    parser.add_argument('--max-concurrent', '-c', type=int, default=5,
                       help='Maximum concurrent API requests')
    parser.add_argument('--retries', '-r', type=int, default=3,
                       help='Number of retries on failure (429/5xx/timeouts)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Scan files without enrolling (for testing)')
    parser.add_argument('--id-source', choices=['filename', 'parent', 'path'], default='filename',
                       help="How to derive user_id from images: 'filename' (default), 'parent' folder, or relative 'path'")
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint and update progress during run')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint JSON file (default: logs/bulk_enrollment_checkpoint.json)')
    
    args = parser.parse_args()
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Bulk Face Enrollment Script Starting")
    logger.info(f"Target folder: {args.folder}")
    logger.info(f"API URL: {args.api_url}")
    logger.info(f"Max concurrent requests: {args.max_concurrent}")
    logger.info(f"User ID source mode: {args.id_source}")
    
    if args.dry_run:
        logger.info("Running in DRY RUN mode")
    
    try:
        async with BulkEnroller(args.api_url, id_source=args.id_source) as enroller:
            stats = await enroller.process_folder(
                args.folder,
                max_concurrent=args.max_concurrent,
                dry_run=args.dry_run,
                retries=args.retries,
                resume=args.resume,
                checkpoint_path=args.checkpoint
            )
            
        if not args.dry_run:
            logger.info(f"Enrollment completed! Success rate: {(stats.successful/max(1, stats.processed))*100:.1f}%")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
