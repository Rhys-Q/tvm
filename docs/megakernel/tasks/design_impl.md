# 任务描述
event tensor 论文在docs/megakernel/2604.13327v2.pdf。
tvm tirx 代码路径在：python/tvm/tirx。
你的任务是基于tvm tirx，来实现event tensor。要支持static schedule以及dynamic schedule。demo就拿论文中的raw sum就行。
本地任务是做设计，你需要出一版设计方案文档，放到docs/megakernel/design。
其实有人已经尝试复现event tensor了：https://github.com/zhen8838/handson-polyhedral/blob/main/18_etensor.ipynb。 我们也可以参考它的实现。这个实现的repo我已经拉取下来了，放在：/root/tw/handson-polyhedral。

# 关键设计点
1. event tensor 应该存放在哪里？是global memory 还是shared memory？
2. event tensor需要支持哪些操作？依赖同步方式怎么实现？
3. /root/tw/tvm/docs/megakernel/tasks/fake_example.py 如何支持这个example？ graph_func 对应的是megakernel，它是一个完整的kernel。device_func 怎样抽象？它对应的是一个tile task。需要做好这个设计，可以参考/root/tw/handson-polyhedral/18_etensor.ipynb的设计。

# 设计方案验收标准
1. 设计方案应该是event tensor的完整设计实现方案，不能是半成品。
2. 设计方案能支持/root/tw/tvm/docs/megakernel/tasks/fake_example.py 类似的example的编译运行。
