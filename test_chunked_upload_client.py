"""
Test client for chunked video upload API

This script demonstrates how to upload large videos using the chunked upload endpoints.

Usage:
    python test_chunked_upload_client.py <video_file_path> <trip_id>

Example:
    python test_chunked_upload_client.py ./large_video.mp4 TRIP-123
"""

import os
import sys
import requests
from pathlib import Path


# Configuration
BASE_URL = "http://localhost:8000/api"
CHUNK_SIZE = 7 * 1024 * 1024  # 8 MB


def upload_video_chunked(video_path: str, trip_id: str):
    """
    Upload a video file using chunked upload

    Args:
        video_path: Path to the video file
        trip_id: Trip identifier
    """
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return

    file_size = os.path.getsize(video_path)
    filename = os.path.basename(video_path)
    total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"📹 Video: {filename}")
    print(f"📊 Size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    print(f"🧩 Total chunks: {total_chunks}")
    print(f"🎫 Trip ID: {trip_id}")
    print()

    # Step 1: Initiate upload
    print("Step 1: Initiating chunked upload...")
    try:
        response = requests.post(
            f"{BASE_URL}/chunked-upload/initiate",
            data={
                "tripId": trip_id,
                "filename": filename,
                "totalSize": file_size,
                "lpCrewName": "Test Pilot",
                "lpCrewId": "LP-001"
            }
        )
        response.raise_for_status()
        init_data = response.json()

        upload_id = init_data["uploadId"]
        print(f"✅ Upload session created")
        print(f"   Upload ID: {upload_id}")
        print(f"   Total chunks: {init_data['totalChunks']}")
        print(f"   Chunk size: {init_data['chunkSize']:,} bytes")
        print(f"   Expires at: {init_data['expiresAt']}")
        print()

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to initiate upload: {e}")
        return

    # Step 2: Upload chunks
    print(f"Step 2: Uploading {total_chunks} chunks...")
    with open(video_path, 'rb') as video_file:
        for chunk_index in range(total_chunks):
            # Read chunk
            video_file.seek(chunk_index * CHUNK_SIZE)
            chunk_data = video_file.read(CHUNK_SIZE)

            # Upload chunk with retry logic
            max_retries = 3
            retry_count = 0
            uploaded = False

            while not uploaded and retry_count < max_retries:
                try:
                    response = requests.post(
                        f"{BASE_URL}/chunked-upload/chunk",
                        data={
                            "uploadId": upload_id,
                            "chunkIndex": chunk_index
                        },
                        files={
                            "chunk": (f"chunk_{chunk_index}.bin", chunk_data, "application/octet-stream")
                        }
                    )
                    response.raise_for_status()
                    chunk_data_response = response.json()

                    uploaded = True
                    progress = (chunk_data_response['receivedChunks'] / chunk_data_response['totalChunks']) * 100
                    print(
                        f"   ✅ Chunk {chunk_index + 1}/{total_chunks} uploaded "
                        f"({progress:.1f}% complete)"
                    )

                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"   ❌ Failed to upload chunk {chunk_index} after {max_retries} retries: {e}")
                        return
                    print(f"   ⚠️  Retry {retry_count}/{max_retries} for chunk {chunk_index}")

    print()

    # Step 3: Finalize upload
    print("Step 3: Finalizing upload and starting video processing...")
    try:
        response = requests.post(
            f"{BASE_URL}/chunked-upload/finalize",
            data={
                "uploadId": upload_id,
                "useMockDetection": "false",
                "useMultiprocessing": "true",
                "saveClips": "false"
            }
        )
        response.raise_for_status()
        result = response.json()

        print(f"✅ Upload and processing complete!")
        print(f"   Status: {result['status']}")
        print(f"   Message: {result['message']}")
        print(f"   Trip ID: {result['tripId']}")
        print(f"   Activities detected: {result['activitiesCount']}")
        print(f"   Processing time: {result.get('processingTime', 0):.2f}s")
        print(f"   Output directory: {result['runDirectory']}")
        print(f"   Activities JSON: {result['activitiesJsonPath']}")
        print()
        print("🎉 Success! Video uploaded and processed successfully.")

    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to finalize upload: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print("Usage: python test_chunked_upload_client.py <video_file_path> <trip_id>")
        print()
        print("Example:")
        print("  python test_chunked_upload_client.py ./large_video.mp4 TRIP-123")
        sys.exit(1)

    video_path = sys.argv[1]
    trip_id = sys.argv[2]

    print("=" * 70)
    print("Chunked Video Upload Test Client")
    print("=" * 70)
    print()

    upload_video_chunked(video_path, trip_id)


if __name__ == "__main__":
    main()
python test_chunked_upload_client.py /Users/satishvanga/Documents/cvvr/bagpack.mp4 TRIP-123