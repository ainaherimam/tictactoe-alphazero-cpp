#!/usr/bin/env python3
"""
Integration Test for Shared Memory Inference Pipeline

This test simulates the complete workflow:
1. C++ worker submitting requests to shared memory
2. Python batcher collecting requests
3. Mock inference processing
4. Responses being written back
5. C++ worker reading responses

This validates the entire IPC protocol without needing actual C++ code.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path if running standalone
sys.path.insert(0, str(Path(__file__).parent))

from shared_memory_interface import SharedMemoryInterface
from batcher import InferenceBatcher
from shared_memory_protocol import (
    JobState,
    INPUT_SIZE,
    POLICY_SIZE,
    INPUT_CHANNELS,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    verify_sizes
)


class MockInferenceEngine:
    """Mock inference engine for testing"""
    
    def __init__(self):
        self.inference_count = 0
    
    def run_inference(self, board_states, legal_masks):
        """
        Mock inference that returns dummy policy and value.
        
        Args:
            board_states: [B, C, H, W]
            legal_masks: [B, POLICY_SIZE]
            
        Returns:
            policy: [B, POLICY_SIZE]
            values: [B]
        """
        batch_size = board_states.shape[0]
        self.inference_count += batch_size
        
        # Generate mock policy (uniform over legal moves)
        policy = legal_masks / legal_masks.sum(axis=1, keepdims=True)
        
        # Generate mock values (based on mean of board state)
        values = np.tanh(board_states.mean(axis=(1, 2, 3)))
        
        # Add small processing delay to simulate real inference
        time.sleep(0.001 * batch_size)  # 1ms per sample
        
        return policy, values


class MockCppWorker:
    """Simulates a C++ worker thread submitting requests"""
    
    def __init__(self, shm: SharedMemoryInterface, worker_id: int):
        self.shm = shm
        self.worker_id = worker_id
        self.next_job_id = worker_id * 10000  # Offset per worker
        self.requests_submitted = 0
        self.responses_received = 0
    
    def submit_request(self) -> tuple:
        """
        Submit a request to shared memory (simulating C++ behavior).
        
        Returns:
            (slot, job_id) if successful, None if no slots available
        """
        # Find a FREE slot
        slot = self._find_free_slot()
        if slot is None:
            return None
        
        # Generate job ID
        job_id = self.next_job_id
        self.next_job_id += 1
        
        # Generate random board state and legal mask
        board_state = np.random.randn(INPUT_SIZE).astype(np.float32)
        legal_mask = np.random.randint(0, 2, POLICY_SIZE).astype(np.float32)
        
        # Ensure at least one legal move
        if legal_mask.sum() == 0:
            legal_mask[0] = 1.0
        
        # Write request to shared memory (simulating C++ atomic operations)
        req = self.shm.buffer.requests[slot]
        
        # Mark as WRITING
        req.state.value = JobState.WRITING
        
        # Write data
        req.job_id.value = job_id
        for i in range(INPUT_SIZE):
            req.board_state[i] = board_state[i]
        for i in range(POLICY_SIZE):
            req.legal_mask[i] = legal_mask[i]
        
        # Mark as READY (this signals Python to process it)
        req.state.value = JobState.READY
        
        # Update stats
        self.shm.buffer.total_requests_submitted.value += 1
        self.requests_submitted += 1
        
        return (slot, job_id)
    
    def _find_free_slot(self) -> int:
        """Find a FREE slot (simulating C++ search)"""
        for slot in range(256):  # MAX_BATCH_SIZE
            if self.shm.buffer.requests[slot].state.value == JobState.FREE:
                return slot
        return None
    
    def wait_for_response(self, slot: int, job_id: int, timeout: float = 5.0) -> dict:
        """
        Wait for response (simulating C++ spin-wait).
        
        Returns:
            dict with 'policy', 'value', 'success'
        """
        start = time.time()
        
        while time.time() - start < timeout:
            req = self.shm.buffer.requests[slot]
            
            # Check if DONE
            if req.state.value == JobState.DONE:
                # Read response
                resp = self.shm.buffer.responses[slot]
                
                # Verify job ID
                if resp.job_id.value != job_id:
                    return {
                        'success': False,
                        'error': f'Job ID mismatch: expected {job_id}, got {resp.job_id.value}'
                    }
                
                # Copy response data
                policy = np.array([resp.policy[i] for i in range(POLICY_SIZE)])
                value = resp.value
                
                # Mark slot as FREE (C++ would do this)
                req.state.value = JobState.FREE
                
                self.responses_received += 1
                
                return {
                    'success': True,
                    'policy': policy,
                    'value': value,
                    'latency': time.time() - start
                }
            
            # Small sleep to avoid burning CPU
            time.sleep(0.0001)  # 0.1ms
        
        return {
            'success': False,
            'error': f'Timeout waiting for response (slot={slot}, job_id={job_id})'
        }


def run_test_suite():
    """Run comprehensive integration tests"""
    
    print("\n" + "=" * 70)
    print("SHARED MEMORY INTEGRATION TEST")
    print("=" * 70)
    
    # Verify protocol sizes
    print("\n[1/6] Verifying protocol sizes...")
    if not verify_sizes():
        print("❌ Protocol size verification failed!")
        return False
    print("✅ Protocol sizes verified")
    
    # Create shared memory
    print("\n[2/6] Creating shared memory...")
    shm = SharedMemoryInterface("test_integration_shm", create=True)
    print("✅ Shared memory created")
    
    # Create batcher
    print("\n[3/6] Creating batcher...")
    batcher = InferenceBatcher(
        max_batch_size=8,
        max_wait_ms=10.0
    )
    print("✅ Batcher created")
    
    # Create mock inference engine
    print("\n[4/6] Creating mock inference engine...")
    engine = MockInferenceEngine()
    print("✅ Mock engine ready")
    
    # Mark server as ready
    shm.set_server_ready(True)
    
    print("\n" + "=" * 70)
    print("TEST 1: Single Request Workflow")
    print("=" * 70)
    
    # Create mock C++ worker
    worker = MockCppWorker(shm, worker_id=1)
    
    # Submit a request
    print("\n[Worker] Submitting request...")
    result = worker.submit_request()
    if result is None:
        print("❌ Failed to submit request (no free slots)")
        shm.cleanup(unlink=True)
        return False
    
    slot, job_id = result
    print(f"✅ Request submitted: slot={slot}, job_id={job_id}")
    
    # Collect batch (server side)
    print("\n[Server] Collecting batch...")
    batch = batcher.collect_batch(shm)
    
    if batch.is_empty():
        print("❌ Batch is empty!")
        shm.cleanup(unlink=True)
        return False
    
    print(f"✅ Batch collected: {len(batch)} requests")
    print(f"   Slots: {batch.slots}")
    print(f"   Job IDs: {batch.job_ids}")
    
    # Reshape for inference
    print("\n[Server] Reshaping for model...")
    board_states, legal_masks = batch.reshape_for_model()
    print(f"✅ Reshaped: board_states={board_states.shape}, legal_masks={legal_masks.shape}")
    
    # Run mock inference
    print("\n[Server] Running inference...")
    policy, values = engine.run_inference(board_states, legal_masks)
    print(f"✅ Inference complete: policy={policy.shape}, values={values.shape}")
    
    # Write responses
    print("\n[Server] Writing responses...")
    for i in range(len(batch)):
        shm.write_response(
            slot=batch.slots[i],
            job_id=batch.job_ids[i],
            policy=policy[i],
            value=float(values[i])
        )
    print(f"✅ Responses written for {len(batch)} requests")
    
    # Wait for response (worker side)
    print("\n[Worker] Waiting for response...")
    response = worker.wait_for_response(slot, job_id)
    
    if not response['success']:
        print(f"❌ Failed to get response: {response.get('error', 'Unknown error')}")
        shm.cleanup(unlink=True)
        return False
    
    print(f"✅ Response received!")
    print(f"   Latency: {response['latency']*1000:.2f}ms")
    print(f"   Value: {response['value']:.4f}")
    print(f"   Policy sum: {response['policy'].sum():.4f}")
    print(f"   Policy top-3: {np.sort(response['policy'])[-3:]}")
    
    print("\n" + "=" * 70)
    print("TEST 2: Multiple Requests (Batching)")
    print("=" * 70)
    
    # Submit multiple requests
    num_requests = 5
    pending_requests = []
    
    print(f"\n[Worker] Submitting {num_requests} requests...")
    for i in range(num_requests):
        result = worker.submit_request()
        if result is None:
            print(f"⚠️  Failed to submit request {i+1}")
            continue
        pending_requests.append(result)
    
    print(f"✅ Submitted {len(pending_requests)} requests")
    
    # Collect batch
    print("\n[Server] Collecting batch...")
    batch = batcher.collect_batch(shm)
    print(f"✅ Batch collected: {len(batch)} requests")
    
    # Process batch
    print("\n[Server] Processing batch...")
    board_states, legal_masks = batch.reshape_for_model()
    policy, values = engine.run_inference(board_states, legal_masks)
    
    for i in range(len(batch)):
        shm.write_response(
            slot=batch.slots[i],
            job_id=batch.job_ids[i],
            policy=policy[i],
            value=float(values[i])
        )
    print(f"✅ Batch processed")
    
    # Wait for all responses
    print("\n[Worker] Waiting for responses...")
    successful = 0
    failed = 0
    latencies = []
    
    for slot, job_id in pending_requests:
        response = worker.wait_for_response(slot, job_id)
        if response['success']:
            successful += 1
            latencies.append(response['latency'])
        else:
            failed += 1
            print(f"   ❌ {response.get('error', 'Unknown error')}")
    
    print(f"✅ Responses: {successful} successful, {failed} failed")
    if latencies:
        print(f"   Avg latency: {np.mean(latencies)*1000:.2f}ms")
        print(f"   Max latency: {np.max(latencies)*1000:.2f}ms")
    
    print("\n" + "=" * 70)
    print("TEST 3: Concurrent Workers")
    print("=" * 70)
    
    # Create multiple workers
    workers = [MockCppWorker(shm, worker_id=i) for i in range(3)]
    all_pending = []
    
    print(f"\n[Workers] {len(workers)} workers submitting requests...")
    for worker_idx, worker in enumerate(workers):
        for i in range(3):  # Each worker submits 3 requests
            result = worker.submit_request()
            if result:
                all_pending.append((worker_idx, result[0], result[1]))
    
    print(f"✅ Total requests submitted: {len(all_pending)}")
    
    # Process in batches
    print("\n[Server] Processing all requests...")
    processed = 0
    
    while processed < len(all_pending):
        batch = batcher.collect_batch(shm)
        if batch.is_empty():
            break
        
        board_states, legal_masks = batch.reshape_for_model()
        policy, values = engine.run_inference(board_states, legal_masks)
        
        for i in range(len(batch)):
            shm.write_response(
                slot=batch.slots[i],
                job_id=batch.job_ids[i],
                policy=policy[i],
                value=float(values[i])
            )
        
        processed += len(batch)
        print(f"   Processed batch of {len(batch)} (total: {processed}/{len(all_pending)})")
    
    print(f"✅ All requests processed")
    
    # Verify responses
    print("\n[Workers] Verifying responses...")
    all_successful = True
    for worker_idx, slot, job_id in all_pending:
        response = workers[worker_idx].wait_for_response(slot, job_id, timeout=1.0)
        if not response['success']:
            print(f"   ❌ Worker {worker_idx}: {response.get('error')}")
            all_successful = False
    
    if all_successful:
        print(f"✅ All responses verified")
    
    print("\n" + "=" * 70)
    print("TEST 4: Statistics and State")
    print("=" * 70)
    
    # Print statistics
    print("\n[Server] Buffer statistics:")
    stats = shm.get_stats()
    for key, value in stats.items():
        print(f"   {key:20}: {value}")
    
    print("\n[Server] Batcher statistics:")
    batcher_stats = batcher.get_stats()
    for key, value in batcher_stats.items():
        print(f"   {key:20}: {value}")
    
    print("\n[Workers] Worker statistics:")
    for i, worker in enumerate(workers):
        print(f"   Worker {i}:")
        print(f"      Submitted: {worker.requests_submitted}")
        print(f"      Received:  {worker.responses_received}")
    
    print("\n[Engine] Inference statistics:")
    print(f"   Total inferences: {engine.inference_count}")
    
    # Print buffer state
    print("\n[Server] Buffer state:")
    shm.print_buffer_state()
    
    print("\n" + "=" * 70)
    print("TEST 5: Error Handling")
    print("=" * 70)
    
    # Test writing response to wrong slot
    print("\n[Server] Testing invalid response write...")
    try:
        shm.write_response(
            slot=0,
            job_id=99999,  # Wrong job ID
            policy=np.zeros(POLICY_SIZE),
            value=0.0
        )
        print("❌ Should have raised an error!")
        all_successful = False
    except RuntimeError as e:
        print(f"✅ Correctly raised error: {e}")
    
    # Test invalid policy shape
    print("\n[Server] Testing invalid policy shape...")
    worker = MockCppWorker(shm, worker_id=99)
    result = worker.submit_request()
    if result:
        slot, job_id = result
        batch = batcher.collect_batch(shm)
        
        try:
            shm.write_response(
                slot=slot,
                job_id=job_id,
                policy=np.zeros(POLICY_SIZE + 1),  # Wrong size!
                value=0.0
            )
            print("❌ Should have raised an error!")
            all_successful = False
        except ValueError as e:
            print(f"✅ Correctly raised error: {e}")
            # Clean up the slot
            shm.buffer.requests[slot].state.value = JobState.FREE
    
    print("\n" + "=" * 70)
    print("TEST 6: Cleanup")
    print("=" * 70)
    
    print("\n[Server] Cleaning up...")
    shm.cleanup(unlink=True)
    print("✅ Cleanup complete")
    
    # Final verdict
    print("\n" + "=" * 70)
    if all_successful:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70 + "\n")
    
    return all_successful


def run_stress_test(num_requests: int = 100, num_workers: int = 4):
    """Run a stress test with many concurrent requests"""
    
    print("\n" + "=" * 70)
    print(f"STRESS TEST: {num_requests} requests, {num_workers} workers")
    print("=" * 70)
    
    # Setup
    shm = SharedMemoryInterface("stress_test_shm", create=True)
    batcher = InferenceBatcher(max_batch_size=32, max_wait_ms=5.0)
    engine = MockInferenceEngine()
    shm.set_server_ready(True)
    
    # Create workers and submit requests
    workers = [MockCppWorker(shm, worker_id=i) for i in range(num_workers)]
    all_pending = []
    
    print(f"\nSubmitting {num_requests} requests...")
    start_submit = time.time()
    
    for i in range(num_requests):
        worker = workers[i % num_workers]
        result = worker.submit_request()
        if result:
            all_pending.append((worker, result[0], result[1]))
        
        # Small delay to simulate realistic timing
        if i % 10 == 0:
            time.sleep(0.001)
    
    submit_time = time.time() - start_submit
    print(f"✅ Submitted {len(all_pending)} requests in {submit_time:.2f}s")
    
    # Process all requests
    print(f"\nProcessing requests...")
    start_process = time.time()
    processed = 0
    batches = 0
    
    while processed < len(all_pending):
        batch = batcher.collect_batch(shm)
        if batch.is_empty():
            time.sleep(0.001)
            continue
        
        board_states, legal_masks = batch.reshape_for_model()
        policy, values = engine.run_inference(board_states, legal_masks)
        
        for i in range(len(batch)):
            shm.write_response(
                slot=batch.slots[i],
                job_id=batch.job_ids[i],
                policy=policy[i],
                value=float(values[i])
            )
        
        processed += len(batch)
        batches += 1
    
    process_time = time.time() - start_process
    print(f"✅ Processed {processed} requests in {batches} batches")
    print(f"   Time: {process_time:.2f}s")
    print(f"   Throughput: {processed/process_time:.1f} req/s")
    print(f"   Avg batch size: {processed/batches:.1f}")
    
    # Verify all responses
    print(f"\nVerifying responses...")
    start_verify = time.time()
    successful = 0
    failed = 0
    
    for worker, slot, job_id in all_pending:
        response = worker.wait_for_response(slot, job_id, timeout=5.0)
        if response['success']:
            successful += 1
        else:
            failed += 1
    
    verify_time = time.time() - start_verify
    print(f"✅ Verification complete: {successful} successful, {failed} failed")
    print(f"   Time: {verify_time:.2f}s")
    
    # Cleanup
    shm.cleanup(unlink=True)
    
    total_time = time.time() - start_submit
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"End-to-end throughput: {len(all_pending)/total_time:.1f} req/s")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Integration test for shared memory inference")
    parser.add_argument('--stress', action='store_true', help='Run stress test')
    parser.add_argument('--num-requests', type=int, default=100, help='Number of requests for stress test')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for stress test')
    
    args = parser.parse_args()
    
    if args.stress:
        run_stress_test(args.num_requests, args.num_workers)
    else:
        success = run_test_suite()
        sys.exit(0 if success else 1)