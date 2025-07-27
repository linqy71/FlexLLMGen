# Important token Indentification

还没跑通版0.0

## OptLM初始化

增加初始化RadixTree和ChunkPool

## generate

在batch_size = 1 && num_bathes = 1的情况下运行。

1.准备好输入，进入generate函数。

2.查询当前输入的公共前缀token的kv_ptr。将字段放入Task中。

3.通过原先就有的set_task，使得每一层都能访问到kv_ptr。新增一个set_chunk_pool函数，让每一层都能使用chunk_pool

4.进入generation_loop_overlap_single_batch函数。进入load_cache函数

5.原先load_cache函数中i==0的时候return。将其注释掉。进入注意力层的load_cache函数。

6.如果有公共前缀的话，就调用load_probe_cache加载探测头的K缓存。这一步需要生成一个在计算设备上的、包含所需数据的张量放到cache_read_buf里面。

7.->compute_layer -> SelfAttention.forward。prefill阶段从cache_read_buf取出探测头的K缓存，用于获取重要token。get_important_token_idx返回重要token在inputs序列中的下标。通过get_prefix_kv获取这些token在所有头上的KV缓存。

8.get_prefix_kv中根据重要token在inputs中的下标，查询Task中的kv_ptr得到缓存所在位置，然后从chunk_pool中复制进来。

9.复制完成得到前缀KV缓存后返回SelfAttention.forward中mha_prefill计算。计算没有被缓存过的(不在公共前缀中的token)新增的k_new,v_new向量。返回拼接后的kv_cache，记录该层prefill阶段的cache长度`self.prefix_cache_shape`（因为get_important_token_idx函数中与threshold比较，有两种情况）将k_cache，v_cache暂存到cache_write_buf中。

10.compute_layer结束后OptLM.store_cache。将cache_write_buf中的向量复制到cache_home中。prefill阶段时，将取出的kv向量整个存储到cache_home中。

11.decoding阶段。load_cache从cache_home中取出（prefill阶段的缓存 + 后续生成的缓存）拼接得到的KV缓存。

12.SelfAttention.forward中，mha_gen需要新增参数pos标识当前KV缓存第一维的长度。后续正常计算(但是没有用mask)，返回新生成的kv_new。存入cache_write_buf

13.OptLM.store_cache部分。decode阶段是将新生成的缓存拼接到原先cache_home缓存后面。通过记录的prefill阶段的cache维度+已生成的token数得到要复制的位置

14.生成完成后，在OptLM.store_prefix_cache中持久化新得到的KV缓存，并向RadixTree中插入。

