import asyncio
import httpx
import time
import statistics
import sys

BASE_URL = "http://localhost:8000"
CONCURRENT_USERS = 5
REQUESTS_PER_USER = 3

async def simulate_user(user_id: int):
    results = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(REQUESTS_PER_USER):
            start = time.perf_counter()
            query = f"Test request {i} from user {user_id}. What can you do?"
            try:
                response = await client.post(
                    f"{BASE_URL}/api/query",
                    json={"user_id": f"bot-{user_id}", "query": query, "top_k": 2}
                )
                latency = (time.perf_counter() - start) * 1000
                results.append({
                    "status": response.status_code,
                    "latency": latency,
                    "valid": response.json().get("valid") if response.status_code == 200 else False
                })
                print(f"User {user_id} - Req {i}: {response.status_code} ({int(latency)}ms)")
            except Exception as e:
                print(f"User {user_id} - Req {i}: FAILED - {e}")
                results.append({"status": 500, "latency": 0, "valid": False})
            
            await asyncio.sleep(0.5) # Slight stagger
    return results

async def main():
    print(f"Starting load test on {BASE_URL}")
    print(f"Users: {CONCURRENT_USERS}, Requests/User: {REQUESTS_PER_USER}")
    
    start_time = time.perf_counter()
    tasks = [simulate_user(i) for i in range(CONCURRENT_USERS)]
    all_results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time
    
    flat_results = [res for user_res in all_results for res in user_res]
    latencies = [r["latency"] for r in flat_results if r["status"] == 200]
    success_count = sum(1 for r in flat_results if r["status"] == 200)
    
    print("\n" + "="*40)
    print("LOAD TEST RESULTS")
    print("="*40)
    print(f"Total Requests: {len(flat_results)}")
    print(f"Success Count:  {success_count}")
    print(f"Total Time:     {total_time:.2f}s")
    print(f"Throughput:     {len(flat_results)/total_time:.2f} req/s")
    
    if latencies:
        print(f"Avg Latency:    {statistics.mean(latencies):.2f}ms")
        print(f"P95 Latency:    {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
    
    # Check metrics endpoint
    async with httpx.AsyncClient() as client:
        try:
            m_resp = await client.get(f"{BASE_URL}/api/health") # Reuse health or check if there's a metrics endpoint
            # In our system metrics is internal but we could expose it. 
            # For now, we've verified the record_request logic.
            pass
        except:
            pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        CONCURRENT_USERS = int(sys.argv[1])
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
