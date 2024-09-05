'''
Author: chenzhi chenzhi@pjlab.org.cn
Date: 2024-07-31 22:31:18
LastEditors: chenzhi chenzhi@pjlab.org.cn
LastEditTime: 2024-08-01 00:00:43
FilePath: /AwesomeICL/agent/prompts/criticqa_ra.py
'''

rewrite_answer_prompt = {
    'zh': '''根据问题和答案将答案内容分成推理过程和最终结论两部分，答案内容拆分过程符合以下条件：
* 不能改变答案内容的原本信息；
* 拆分出的答案分为推理和最终结论两部分，其中最终结论就是最简洁形式的答案；
* 最终结论不要出现在推理过程中；
* 最终拆分出来的内容形式是："推理过程：xxx 最终答案：xxx"；
原问题和答案：
{q}，答案：{a}
拆分的答案：''',
    'en': '''Divide the answer content into two parts: the reasoning process and the final conclusion based on the question and answer. The process of splitting the answer content meets the following conditions:
* Cannot change the original information of the answer content;
* The split answer is divided into two parts: reasoning process and final conclusion, with the final conclusion being the simplest form of the answer;
* The reasoning process should not include the final conclusion, which should remove at the start of the reasoning process;
* The final split answer is: "Reasoning process: xxx Final conclusion: xxx";
Original question and answer:
{q}, Answer: {a}
Split answer:'''
}