import asyncio
import httpx
import aiohttp
from pprint import pprint
from time import time

limits = httpx.Limits(max_connections=5000, max_keepalive_connections=5000)

urls = [f"https://pokeapi.co/api/v2/pokemon/{i}" for i in range(1, 1001)] * 5
semaphore = asyncio.Semaphore(2000)
async def get_posts(url: str, client: httpx.AsyncClient):
    async with semaphore:
        response = await client.get(url)
        # pprint(f"Request to {url} - Status: {response.status_code}")
    # pprint(response.status_code)
    return response.json()

max_req_per_sec = 1000  # e.g., 10 requests per second
delay = 1 / max_req_per_sec

async def main():
    start_time = time()
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = []
        for i, url in enumerate(urls):
            task = asyncio.create_task(get_posts(client=client, url=url))
            tasks.append(task)

            # Rate-limiting: Ensure that we respect max_req_per_sec
            if (i + 1) % max_req_per_sec == 0:  # After every `max_req_per_sec` requests
                await asyncio.sleep(1)  # Wait for the necessary time to stay within rate limit
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    print(len(responses))

    elapsed_time = time() - start_time
    print(f"Total requests sent: {len(responses)}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    # Calculate the rate of requests per second
    request_rate = len(responses) / elapsed_time
    print(f"Request rate: {request_rate:.2f} requests per second")
    pprint(responses[0].keys())
asyncio.run(main())