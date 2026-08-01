# 掌握 Python 异步编程：从基础到生产实践

## 面对 I/O 密集型任务，何时选择 async/await 而非线程或多进程

判断方式很直接：异步只在 I/O 密集型工作中表现出色。能用 asyncio 时就用它，必须使用线程时再选择 threading 或 concurrent.futures——这条经验法则可以指导你的选择。

对于 I/O 密集型任务，异步 I/O 通常比多线程更快，尤其是在管理大量并发任务时，因为它避免了线程管理的开销。在 Linux 上，每个操作系统线程默认消耗 8MB 内存。这意味着仅栈空间一项，1,000 个线程就会消耗 8GB 内存。相比之下，asyncio 可以在单个线程中运行 100,000 个协程，且内存开销很小——每个协程仅占用约 4KB。

它的杀手级优势是协作式并发。协程遇到 `await` 时会主动将控制权交还事件循环，使成千上万个其他协程能在它等待 I/O 时继续推进。

CPU 密集型工作需要另一种工具。Python 的全局解释器锁（GIL）使线程无法实现真正的并行，因此异步和线程对计算密集型任务同样无效。如果要执行繁重计算，请使用 `multiprocessing`；对于其他涉及 I/O 延迟的工作，则应选择异步。

## 使用 asyncio 事件循环和协程构建第一个异步应用

Python 异步 I/O 的核心构件是可等待对象（通常是协程），事件循环会异步地调度和执行它们。这种编程模型让你能在单个执行线程中高效管理多个 I/O 密集型任务。

从一个简单示例开始：

```python
import asyncio

async def fetch_data(name):
    print(f"开始 {name}")
    await asyncio.sleep(1)  # 模拟 I/O
    print(f"完成 {name}")
    return f"来自 {name} 的数据"

async def main():
    # 并发运行三个协程
    results = await asyncio.gather(
        fetch_data("API-1"),
        fetch_data("API-2"),
        fetch_data("API-3")
    )
    print(results)

asyncio.run(main())
```

这段代码总共只需约 1 秒，而不是 3 秒。当一个任务在事件循环中运行时，同一线程中的其他任务无法运行。当该任务执行 `await` 表达式时，它会被挂起，事件循环转而执行下一个任务。

事件循环是一个单线程调度器。它维护着就绪任务队列和 I/O 事件登记表。当代码执行到 `await` 时，事件循环会挂起该协程并运行下一个就绪任务，从而无需承担线程开销便可实现并发。

始终使用 `asyncio.run()` 启动主协程。此函数会自动创建事件循环、运行协程，并在完成后正确清理资源。

## 在不阻塞执行的情况下处理并发 API 调用和数据库操作

真实世界中的异步编程需要合适的库。编写异步 I/O 代码时，常用的两个工具是用于 HTTP 调用的 aiohttp 库和用于数据库访问的 SQLAlchemy 异步 ORM。

以下是使用 `aiohttp` 并发调用 API 的模式：

```python
import aiohttp
import asyncio

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.json()

async def fetch_all_urls(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

urls = [
    "https://api.example.com/user/1",
    "https://api.example.com/user/2",
    "https://api.example.com/user/3"
]

asyncio.run(fetch_all_urls(urls))
```

复用同一个 `ClientSession` 可启用 HTTP 连接池，让多个请求重复使用同一条 TCP 连接。与每次创建新连接相比，这可以将每个请求的延迟降低 20 至 50 毫秒。切勿为每个请求创建新会话，否则连接池就失去了意义。

对于数据库，请使用 `asyncpg`（PostgreSQL）或 `aiosqlite` 等异步驱动程序。`sqlite3` 或 `psycopg2` 等标准数据库库会在查询期间阻塞事件循环。例如，`asyncpg` 可以处理一万多个并发数据库连接，而同步的 `psycopg2` 则需要一万个线程。

## 调试异步代码：异步函数中的阻塞调用等常见陷阱及其修复方法

最隐蔽的错误是在异步代码中调用阻塞函数。使用 `time.sleep()` 会冻结整个事件循环，阻塞所有并发协程：

```python
async def bad_example():
    import time
    time.sleep(1)  # 阻塞所有任务

async def good_example():
    await asyncio.sleep(1)  # 将控制权交还事件循环
```

如需在异步函数中执行 CPU 密集型工作，可使用 `loop.run_in_executor()` 在线程池中运行阻塞操作：

```python
async def compute_heavy():
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, expensive_calculation)
    return result
```

另一个常见错误是忘记使用 `await`。缺少 await 时，你得到的是协程对象而非返回值。Python 3.7 及以上版本会发出 `RuntimeWarning: coroutine 'function_name' was never awaited` 警告。修复方法始终相同：加上 `await`。

最严重的错误是直接在协程中运行 CPU 密集型操作。一次耗时 100 毫秒的计算会使 10,000 个并发协程在此期间全部冻结。良好的做法是在线程池中运行 CPU 密集型工作。

## 通过连接池、速率限制和正确的错误处理扩展异步应用

生产级异步应用需要防护措施。复用会话即可获得连接池，但速率限制需要显式控制。尽管协程很轻量（每个约 4KB），创建 50,000 个以上的并发任务仍可能耗尽网络连接并压垮目标服务器。

使用 `asyncio.Semaphore` 限制并发操作数：

```python
async def rate_limited_fetch(semaphore, session, url):
    async with semaphore:
        async with session.get(url) as response:
            return await response.json()

async def fetch_with_limits(urls, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [
            rate_limited_fetch(semaphore, session, url)
            for url in urls
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

`return_exceptions=True` 参数至关重要。默认情况下，只要一个任务失败，`asyncio.gather()` 就会取消所有剩余任务。设置 `return_exceptions=True` 后，失败的操作会返回异常，成功的操作则返回结果：

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for i, result in enumerate(results):
    if isinstance(result, Exception):
        logger.error(f"URL {i} 失败：{result}")
    else:
        process(result)
```

对于需要处理数千个并发连接的生产系统，应组合使用以下方法：复用会话以实现连接池；使用 `Semaphore` 实施速率限制（通常每项服务允许 10 至 100 个并发请求）；使用 `gather(..., return_exceptions=True)` 提高韧性；使用异步数据库库避免持久化操作造成阻塞。Instagram 工程团队报告称，采用这些模式后，每个 Python 进程可处理一万多个并发连接。

## 要点总结

- **使用 `asyncio.run()` 作为入口点**，并始终对协程调用使用 `await`——忘记 await 会触发运行时警告，并返回协程对象而不是值。

- **在多个请求之间复用 `aiohttp.ClientSession` 对象**以启用连接池，可将每次调用的请求延迟降低 20 至 50 毫秒。

- **将 CPU 密集型操作封装在 `loop.run_in_executor(None, function)` 中**，以免阻塞事件循环——绝不要在异步函数中直接调用 `time.sleep()` 或执行繁重计算。

- **使用 `asyncio.Semaphore(N)` 实施速率限制**，N 通常设为每个外部服务 10 至 100 个并发操作，以免压垮目标服务。

- **在 `asyncio.gather()` 中始终使用 `return_exceptions=True`**，防止批次中的一个任务失败后取消所有剩余操作。

## 资料来源

- [更快的 Python：async/await 与线程中的并发 | PyCharm 博客](https://blog.jetbrains.com/pycharm/2025/06/concurrency-in-async-await-and-threading/)
- [Python 中 Asyncio 与线程的比较——GeeksforGeeks](https://www.geeksforgeeks.org/python/asyncio-vs-threading-in-python/)
- [使用并发加速 Python 程序——Real Python](https://realpython.com/python-concurrency/)
- [Python asyncio 比线程更好吗？| ProxiesAPI](https://proxiesapi.com/articles/is-asyncio-python-better-than-threading)
- [Python 中的异步编程与线程](https://medium.com/@sanjeets1900/asynchronous-programming-vs-threading-in-python-d59306a853a7)
- [Python 中多进程、线程与 AsyncIO 的比较——Lei Mao 的博客](https://leimao.github.io/blog/Python-Concurrency-High-Level/)
- [在自由线程与异步之间进行选择——Optiver](https://optiver.com/working-at-optiver/career-hub/choosing-between-free-threading-and-async-in-python/)
- [asyncio 相比线程有哪些优势？——Python.org 讨论](https://discuss.python.org/t/what-are-the-advantages-of-asyncio-over-threads/2112)
- [Python asyncio 实战演练——Real Python](https://realpython.com/async-io-python/)
- [为何异步备受青睐？使用线程实现高级控制流](https://emptysqua.re/blog/why-should-async-get-all-the-love/)
- [Python asyncio 实战演练——Real Python](https://realpython.com/async-io-python/)
- [事件循环——Python 3.14.3 文档](https://docs.python.org/3/library/asyncio-eventloop.html)
- [asyncio 概念概览——Python 3.14.3 文档](https://docs.python.org/3/howto/a-conceptual-overview-of-asyncio.html)
- [Asyncio 事件循环教程 | TutorialEdge.net](https://tutorialedge.net/python/concurrency/asyncio-event-loops-tutorial/)
- [理解 Python asyncio：深入事件循环](https://medium.com/delivus/understanding-pythons-asyncio-a-deep-dive-into-the-event-loop-89a6c5acbc84)
- [协程与任务——Python 3.14.3 文档](https://docs.python.org/3/library/asyncio-task.html)
- [使用 asyncio 开发——Python 3.14.3 文档](https://docs.python.org/3/library/asyncio-dev.html)
- [Python/Django AsyncIO 教程：Python 异步编程](https://djangostars.com/blog/asynchronous-programming-in-python-asyncio/)
- [掌握 Python Asyncio：实用指南](https://medium.com/@moraneus/mastering-pythons-asyncio-a-practical-guide-0a673265cf04)
- [Python 异步编程：完整指南 | DataCamp](https://www.datacamp.com/tutorial/python-async-programming)
- [Python 异步并发：比较 asyncio.gather 下的 aiohttp.ClientSession 与 SQLAlchemy AsyncSession](https://levelup.gitconnected.com/async-concurrency-in-python-comparing-aiohttp-clientsession-033c234a4572)
- [使用 AIOHTTP 发起并发请求的实用指南](https://apidog.com/blog/aiohttp-concurrent-request/)
- [使用 asyncio 加速 ETLHelper 的 API 传输](https://britishgeologicalsurvey.github.io/open-source/async-etlhelper-api-transfer/)
- [Python 异步编程——asyncio 与 aiohttp](https://medium.com/@adityakolpe/python-asynchronous-programming-with-asyncio-and-aiohttp-186378526b01)
- [使用 Python AsyncIO 发起并发 HTTP 请求 | LAAC Technology](https://www.laac.dev/blog/concurrent-http-requests-python-asyncio/)
- [在 Python 中使用 aiohttp 和 asyncio 处理大型请求](https://medium.com/@rspatel031/handling-large-requests-with-aiohttp-and-asyncio-in-python-d603b2de5c69)
- [使用 aiohttp 发起并行 HTTP 请求（视频）——Real Python](https://realpython.com/lessons/making-parallel-http-requests-aiohttp/)
- [使用 aiohttp 在 Python 中发起并发请求 | ProxiesAPI](https://proxiesapi.com/articles/making-concurrent-requests-with-aiohttp-in-python)
- [使用 Python 3 和 asyncio 发起并发 HTTP 请求——GitHub](https://gist.github.com/debugtalk/3d26581686b63c28227777569c02cf2c)
- [asyncio——异步 I/O](https://docs.python.org/3/library/asyncio.html)
