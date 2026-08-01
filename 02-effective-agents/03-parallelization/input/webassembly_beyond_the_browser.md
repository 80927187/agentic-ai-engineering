# 超越浏览器的 WebAssembly：释放服务端与边缘计算潜力

WebAssembly 已经走出浏览器沙箱，成为一种生产级服务端技术。Shopify 等公司使用 WASM 运行不受信任的用户脚本，Fastly 通过 WASM 模块处理数百万个边缘请求，Docker 现在也支持运行纯 WebAssembly 而非 Linux 二进制文件的容器。

## 在 Node.js 中使用 Wasmtime 和 Wasmer 运行时运行 WebAssembly 模块

Node.js WebAssembly 领域主要由两种运行时主导，它们各自具有不同的性能特征：

**Wasmtime** 可达到原生执行速度的 85% 至 90%，内存开销为 25MB，非常适合要求符合 WASI 规范的服务端应用。字节码联盟将其作为 WASI 的参考实现进行维护。

**Wasmer** 仅使用 18MB 内存便可达到原生性能的 80% 至 85%，针对 CLI 工具和插件系统中的轻量级嵌入进行了优化。

这种性能差异在大规模应用中十分重要。Shopify 的内部基准测试显示，Wasmtime 在处理 10,000 个并发脚本执行任务时内存压力更低，而 Wasmer 则在需要快速实例化模块的场景中表现出色。

以下是在 Node.js 中运行 CPU 密集型 WASM 的方法：

```javascript
const { WASI } = require('wasi');
const fs = require('fs');
const wasmBuffer = fs.readFileSync('fibonacci.wasm');

const wasi = new WASI({
  version: 'preview1',
  args: process.argv,
  env: process.env,
  preopens: { '/local': '/tmp' }
});

const instance = new WebAssembly.Instance(
  new WebAssembly.Module(wasmBuffer),
  wasi.getImportObject()
);

// 执行 WASM 函数
const fib = instance.exports.fibonacci;
console.log(fib(40)); // 运行速度约为原生速度的 85%
```

这种模式将计算密集型操作隔离开来，同时由 Node.js 处理 I/O、网络和系统集成。

## 使用 WebAssembly 为 AWS Lambda 和 Cloudflare Workers 构建无服务器函数

性能基准测试揭示了不同无服务器 WASM 平台之间的显著差异：

**Cloudflare Workers** 使用 V8 Isolate，可实现低于 5 毫秒的冷启动，并且在第 95 百分位的性能比 Lambda 快 441%。Workers 分布于全球 200 多座城市，非常适合对延迟敏感的应用。

采用自定义 WASM 运行时的 **AWS Lambda** 可与 S3、DynamoDB 和 API Gateway 无缝集成，但仍会受到传统容器冷启动惩罚（100 至 500 毫秒）的影响。

Cloudflare Workers 支持直接编译 Rust：

```rust
use worker::*;

#[event(fetch)]
pub async fn main(req: Request, _env: Env, _ctx: Context) -> Result<Response> {
    let image_data = req.bytes().await?;
    let compressed = compress_image(&image_data)?; // CPU 密集型 WASM
    Response::ok(compressed)
}
```

对于 AWS Lambda，可将 WASM 与自定义运行时打包：

```rust
// 编译为 wasm32-wasi 目标
pub fn lambda_handler(event: LambdaEvent<Value>) -> Result<Value> {
    let data = event.payload["data"].as_str().unwrap();
    let result = process_data(data); // 在 WASM 沙箱中运行
    Ok(json!({"processed": result}))
}
```

需要优化全球延迟时选择 Workers；需要与 AWS 生态系统深度集成时选择 Lambda。

## 使用 WASI（WebAssembly 系统接口）访问文件系统和网络资源

WASI 通过基于能力的安全机制提供受控的系统访问。与传统容器不同，除非明确授权，否则 WASI 模块无法访问任何宿主资源。

**当前 WASI 的限制**：WASIp1 不支持网络和套接字。不过，2024 年初发布的 WASIp2 通过 `wasi-http` 增加了 HTTP 客户端/服务器支持，并通过 `wasi-keyvalue` 增加了键值存储支持。

**安全优势**：访问宿主文件系统必须通过 `preopens` 明确授权。获准访问 `/app` 的 WASM 模块无法读取 `/etc/passwd` 或其他宿主路径：

```javascript
const wasi = new WASI({
  version: 'preview1',
  preopens: {
    '/app': '/var/app',           // WASM 看到 /app，它映射到 /var/app
    '/data': '/mnt/data'          // WASM 看到 /data，它映射到 /mnt/data
  }
});
```

这种能力模型可防止传统应用中常见的一整类目录遍历漏洞。

## 使用 Docker 和 Kubernetes 编排部署 WebAssembly 微服务

使用 WASM 可显著提高容器密度。典型的 Node.js 微服务容器大小为 200 至 500MB，而包含运行时的等效 WASM 模块仅为 10 至 50MB。

在 Docker 中使用 Wasmtime 运行时：

```dockerfile
FROM scratch
COPY --from=wasmtime:latest /usr/bin/wasmtime /wasmtime
COPY api.wasm /app/
ENTRYPOINT ["/wasmtime", "/app/api.wasm"]
```

使用 runwasi 在 Kubernetes 中原生调度 WASM：

```yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: wasmtime
handler: wasmtime
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wasm-api
spec:
  template:
    spec:
      runtimeClassName: wasmtime
      containers:
      - name: api
        image: ghcr.io/myorg/wasm-api:v1.0.0
        resources:
          requests:
            memory: "32Mi"    # 显著低于传统容器
            cpu: "100m"
```

CNCF 报告显示，与传统 Linux 容器相比，WASM 容器在相同硬件上可实现 3 至 5 倍的 Pod 密度。

## 针对图像处理和密码学等 CPU 密集型任务优化 WebAssembly 性能

对于计算密集型操作，WASM 通常可达到原生性能的 70% 至 90%。性能差距来自边界检查和 JIT 编译开销，而非根本性限制。

**关键优化**：在保持状态隔离的同时，让多个请求共享编译结果：

```rust
// 全局引擎和模块（仅编译一次）
lazy_static! {
    static ref ENGINE: Engine = Engine::default();
    static ref MODULE: Module = Module::from_file(&ENGINE, "crypto.wasm").unwrap();
}

// 按请求隔离
fn handle_request(data: &[u8]) -> Result<Vec<u8>> {
    let mut store = Store::new(&ENGINE, ());
    let instance = Instance::new(&mut store, &MODULE, &[])?;
    let hash_fn = instance.get_typed_func::<(i32, i32), i32>(&mut store, "sha256")?;

    // 内存安全：边界检查可防止缓冲区溢出
    let result = hash_fn.call(&mut store, (data.as_ptr() as i32, data.len() as i32))?;
    Ok(extract_result(&mut store, result))
}
```

**性能数据**：基准测试显示，WASM 图像滤镜在保持完全内存安全的同时，运行速度可达到原生速度的 75% 至 85%。对于密码学操作，V8 的共享代码缓存能以极低的编译开销实现每秒数千次哈希运算。

启用燃料计量以防止无限循环：

```rust
let mut config = Config::new();
config.consume_fuel(true);
let mut store = Store::new(&engine, ());
store.fuel_consumed().unwrap(); // 跟踪执行成本
```

## 要点总结

- **对于要求符合 WASI 规范的服务端工作负载，可评估 Wasmtime；对于轻量级嵌入，可评估 Wasmer**——使用自己的 CPU 密集型代码对二者进行基准测试，确定哪一种更适合你的用例。

- **在 Cloudflare Workers 上部署 WASM，以获得低于 5 毫秒的冷启动和全球延迟优化**；只有在需要与 S3、DynamoDB 或其他 AWS 服务深度集成时，才选择采用 WASM 的 AWS Lambda。

- **采用共享 Engine/Module、按请求隔离 Store/Instance 的模式**，在保持内存安全的同时，使高并发场景获得最佳性能。

- **使用 WASI preopens 授予最小限度的文件系统访问权限**，并启用燃料计量以防执行失控——这种基于能力的安全模型可消除一整类漏洞。

- **将 WASM 用于可达到原生性能 70% 至 90% 的纯计算任务**，并让宿主运行时继续处理 I/O 操作——分析热点路径，将 CPU 密集型函数迁移到 WASM 模块。

服务端 WASM 生态系统已经跨越实验阶段。如果需要大规模处理图像、执行密码学操作或运行不受信任的代码，WebAssembly 可以提供生产级性能和传统容器无法媲美的内置安全保障。

## 资料来源

- [WebAssembly 运行时研究综述](https://arxiv.org/html/2404.12621v1)
- [Bonviewpress](https://ojs.bonviewpress.com/index.php/AAES/article/download/4965/1367/29227)
- [GitHub - appcypher/awesome-wasm-runtimes：WebAssembly 运行时列表](https://github.com/appcypher/awesome-wasm-runtimes)
- [wasmtime-demos/nodejs/README.md](https://github.com/bytecodealliance/wasmtime-demos/blob/main/nodejs/README.md)
- [使用 WasmEdge、Wasmtime 和 Wasmer 调用 MongoDB、Kafka 与 Oracle 开发：WASI Cycles](https://medium.com/oracledevs/develop-with-wasmedge-wasmtime-and-wasmer-invoking-mongodb-kafka-and-oracle-wasi-cycles-an-ad2302fe961a)
- [Wasmtime](https://wasmtime.dev/)
- [WASI 与 WebAssembly 组件模型：当前状态](https://eunomia.dev/blog/2025/02/16/wasi-and-the-webassembly-component-model-current-status/)
- [Web 之外：使用 Emscripten 构建独立 WebAssembly 二进制文件](https://v8.dev/blog/emscripten-standalone-wasm)
- [Wasmtime 深入教程](https://wasmruntime.com/en/tutorials/wasmtime)
- [选择 WebAssembly 运行时](https://blog.colinbreck.com/choosing-a-webassembly-run-time/)
- [AWS Lambda 与 Cloudflare Workers 详细比较](https://5ly.co/blog/aws-lambda-vs-cloudflare-workers/)
- [无服务器计算如何提高性能？| Cloudflare](https://www.cloudflare.com/learning/serverless/serverless-performance/)
- [无服务器的兴起：使用 AWS Lambda 和 Cloudflare Workers 驱动现代应用](https://medium.com/@aayush71727/the-rise-of-serverless-powering-modern-apps-with-aws-lambda-and-cloudflare-workers-c044020eff6c)
- [使用 Cloudflare Workers 实现无服务器架构](https://www.smashingmagazine.com/2019/04/cloudflare-workers-serverless/)
- [2026 年最佳 Cloudflare Workers 替代方案](https://northflank.com/blog/best-cloudflare-workers-alternatives)
- [无服务器性能：Cloudflare Workers、Lambda 和 Lambda@Edge](https://blog.cloudflare.com/serverless-performance-comparison-workers-lambda/)
- [AWS Lambda 与 Cloudflare Workers | Upstash 博客](https://upstash.com/blog/aws-lambda-vs-cloudflare-workers)
- [Python Workers 再进化：快速冷启动、软件包和 uv 优先工作流](https://blog.cloudflare.com/python-workers-advancements/)
- [初探 Cloudflare Workers](https://willhamill.com/2019/01/23/taking-a-look-at-cloudflare-workers)
- [Cloudflare Workers 借助 V8 Isolate 和 WebAssembly 实现无容器云计算](https://hub.packtpub.com/cloudflares-workers-enable-containerless-cloud-computing-powered-by-v8-isolates-and-webassembly/)
- [WASI.dev 简介](https://wasi.dev/)
- [WebAssembly 系统接口（WASI）| Node.js v25.6.1 文档](https://nodejs.org/api/wasi.html)
- [WASI 简介](https://wasmbyexample.dev/examples/wasi-introduction/wasi-introduction.all.en-us)
- [GitHub - WebAssembly/WASI：WebAssembly 系统接口](https://github.com/WebAssembly/WASI)
- [GitHub - WebAssembly/wasi-filesystem：文件系统 API](https://github.com/WebAssembly/wasi-filesystem)
- [什么是 WASI？| Fastly](https://www.fastly.com/learning/serverless/what-is-wasi)
- [WASI：一种新型系统接口——InfoQ](https://www.infoq.com/presentations/wasi-system-interface/)
- [WASI 目前处于什么状态？](https://www.fermyon.com/blog/whats-the-state-of-wasi)
- [Wasm、WASI、Wagi：它们是什么？](https://www.fermyon.com/blog/wasm-wasi-wagi)
- [WebAssembly、WASI 与组件模型](https://www.fermyon.com/blog/webassembly-wasi-and-the-component-model)
