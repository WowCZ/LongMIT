'''
Description: 
Version: 1.0
Autor: Zhi Chen
Date: 2024-07-17 12:31:50
LastEditors: chenzhi chenzhi@pjlab.org.cn
LastEditTime: 2024-07-21 17:33:52
'''

extract_question_prompt = {
    'zh': '''文档内容如下：
{chunk}
结合上面文档提取出文档中包含的问题，摘取出的问题应该符合以下条件：
* 提取出的问题不能任何图片信息；
* 提取的问题中不能有任何指代信息；
* 保证提取问题的完整性，如果是选择题需要提供对应的选项信息，去掉换行符，与题目主体放在一个题目内；
* 如果文档中包含数字、时间、人物、地点等概念的话，必须提取这部分内容形成的问题；
* 提取出的问题用可解析的列表的形式表示，例如：["xxx", "xxx"]，如果没有有价值的问题就直接输出[]；
* 尽可能多提取有价值的问题，但是不能包括重复问题；
* 提取出的问题不超过三个；
提取出的问题：''',
    'en': '''The document content is as follows:
{chunk}
Extract the questions contained in the above document, and the extracted questions should meet the following conditions:
* No pictorial information should be included in the extracted questions;
* No referential information should be included in the extracted questions;
* Ensure the completeness of the extracted questions; if they are multiple-choice questions, provide corresponding option information, remove line breaks, and place the question body in a single question;
* If the document contains concepts such as numbers, time, people, or places, questions that involve this information must be extracted;
* The extracted questions should be presented in a parseable list format, such as ["xxx", "xxx"]. If there are no valuable questions, output an empty list [];
* Try to extract as many valuable questions as possible, but do not include duplicate questions;
* Extract no more than three questions;
Extracted questions:'''
}

filter_question_prompt = {
    'zh': '''判断下面给出的问题是否存在指代信息，如果存在请给出存在指代的位置，并给出得分1；如果不存在给出得分0。
问题如下：
{question}
是否存在指代并给出得分：''',
    'en': '''Determine if there is any referential information for the given question. If so, please provide the location where the referential information exists and give a score of 1; If there is no score given 0.
The question is as follows:
{question}
Is there a reference and a score given:'''
}

generate_answer_prompt = {
    'zh': '''结合文档内容和对应问题生成回答，生成的回答须符合以下条件：
* 结合文档中的内容进行回复；
* 如果文档中没有对应问题回答，请结合自身知识进行回复；
* 如果问题是关于数字、时间、人物、地点等事实性问题时，请直接给出问题回答；
文档内容如下：
{chunk}
问题如下：
{question}
回答：''',
    'en': '''Generate answers to a given series of questions based on the content of the document, which must meet the following conditions:
* Respond based on the content in the document;
* If there is no corresponding answer to the question in the document, please reply based on your own knowledge;
* If the question is about factual issues such as numbers, time, people, places, etc., please provide the answer directly, and different question and answer pairs should be distinguished by line breaks;
The document content is as follows:
{chunk}
The problems are as follows:
{question}
The corresponding answers are as follows:'''
}

simplify_qa_prompt = {
    'zh': '''根据问题和答案生成简洁的答案内容，简洁答案符合以下条件：
* 不包含任何答案信息的任何说明问题；
* 如果答案是人名、时间、地点等实体信息，直接给出实体内容即可；
* 如果答案是判定结果，直接给出是或者否即可；
原问题答案：
问题：{q}，答案：{a}
简化回答：''',
    'en': '''Generate concise answer content based on the question and answer, and concise answers meet the following conditions:
* Any explanatory questions that do not contain any answer information;
* If the answer is physical information such as person name, time, location, etc., simply provide the physical content;
* If the answer is the judgment result, simply provide yes or no;
Answer to the original question:
Question: {q}, Answer: {a}
Simplified answer:'''
}

merge_qa_prompt = {
    'zh': '''根据给定的两个问题回答对，合成一个符合真实场景的问题回答对，合成的问题回答对应该符合以下条件：
* 如果两个问题和回答都与时间相关，可以合成比较两个事件发生先后顺序的比较问题；
* 如果两个问题和回答都与人物相关，可以合成判定那个人物更加符合合成问题描述的问题回答对；
* 合成的回答要给出对应的推理过程，合成的回答尽可能多利用给定两个回答里面的内容；
* 不要随意更改两个问题和回答的原本信息；
* 生成的问题和回答严格用{{"问题": xxx, "回答": xxx}}的jsonl形式输出，每个问答对中不要有任何换行符；
两个问题回答对如下：
{qa1}
{qa2}
合成的问题回答对：''',
    'en': '''Based on the given two question answer pairs, synthesize up to one question answer pair that match the real scenario. The synthesized question answer pair should meet the following conditions:
* If both questions and answers are time related, a comparative question can be synthesized to compare the order in which two events occur;
* If both questions and answers are related to the character, it can be synthesized to determine which character better fits the description of the composite question;
* The synthesized answer should provide the corresponding reasoning process, and the synthesized answer should make as much use of the content in the given two answers as possible;
* Do not arbitrarily change the original information of two questions and answers;
* The generated questions and answers are strictly output in JSONL format using {{"question": xxx, "answer": xxx}}. Synthesized question answer pair should not have any line breaks;
The correct answers to two questions are as follows:
{qa1}
{qa2}
The synthesized question answer pair is:'''
}