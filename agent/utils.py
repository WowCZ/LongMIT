'''
Description: 
Version: 1.0
Autor: Zhi Chen
Date: 2024-07-17 12:04:31
LastEditors: chenzhi chenzhi@pjlab.org.cn
LastEditTime: 2024-08-06 16:13:43
'''

import os, sys
sys.path.append(os.getcwd())

import json
import re
import math
import time
import random
import logging
from openai import OpenAI
from typing import List
from InternEmbedding.embedding.eval.metrics import matrix_cosine_similarity

logger = logging.getLogger(__name__)

def cluster_qa_pairs(sqa_pairs, similar_matrix, topk):
    '''
    description: 
    return {*}
    '''    
    sqa_pair_lens = [len(sqa) for sqa in sqa_pairs if len(sqa) > 0]
    
    indoc_sqa_ids = []
    for si, pair_len in enumerate(sqa_pair_lens):
        if si == 0:
            indoc_sqa_ids.append(list(range(pair_len)))
        else:
            indoc_sqa_ids.append([sum(sqa_pair_lens[:si]) + l for l in range(pair_len)])
    
    indoc_qa_map = dict()
    for mi, sqa_ids in enumerate(indoc_sqa_ids):
        for qi in sqa_ids:
            indoc_qa_map[qi] = mi

    sqa_pair_knns = []
    for i, qa_knn in enumerate(similar_matrix):
        qa_knn = [(s, si) for si, s in enumerate(qa_knn)]
        qa_knn = qa_knn[i:]
        qa_knn = sorted(qa_knn, key=lambda x:x[0], reverse=True)
        qa_knn = [si for _, si in qa_knn]
        sqa_pair_knns.append(qa_knn)

    qa_pair_map = []
    for qi, knn_ids in enumerate(sqa_pair_knns):
        indoc_ids = indoc_sqa_ids[indoc_qa_map[qi]]
        crossdoc_ids = [id for id in knn_ids if id not in indoc_ids]
        qa_pair_map.append(
            {
                'indoc_ids': indoc_ids[1:topk+1],
                'crossdoc_ids': crossdoc_ids[:topk]
            }
        )
    return qa_pair_map


def calc_similarity_matrix(qa_pairs, embedder, embedder_cfg, lang):
    format_qa_pairs = []
    if lang == 'en':
        qk = 'question'
        ak = 'answer'
    else:
        qk = '问题'
        ak = '回答'

    for doc_qa_pairs in qa_pairs:
        for qa in doc_qa_pairs:
            try:
                format_qa_pairs.append(qa[qk] + '\n' + qa[ak])
            except:
                if qk == 'question':
                    qk = '问题'
                    ak = '回答'
                else:
                    qk = 'question'
                    ak = 'answer'
                format_qa_pairs.append(qa[qk] + '\n' + qa[ak])
    
    global_similarity_matrix = embedder.batch_encode(format_qa_pairs, embedder_cfg.encode_batch_size)
    global_similarity_matrix = matrix_cosine_similarity(global_similarity_matrix)

    return global_similarity_matrix


def get_llm_response(question: str, model: str='internlm2-chat-20b', api_base: str='http://172.28.0.81:20241/v1'):

    client = OpenAI(
        api_key=random.choice("EMPTY"), base_url=api_base
    )

    error = None
    num_failures = 0
    while num_failures < 5:
        try:
            response =  client.chat.completions.create(
                    model=model, 
                    messages=[{"role": "user", "content": question}], 
                    max_tokens=4096, 
                    temperature=1.2, 
                    top_p=0.1, 
                    n=1
                )
            # usage = response.usage.completion_tokens
            text = response.choices[0].message.content

            return text
        except Exception as error:
            logger.warning(error)
            num_failures += 1
            time.sleep(5)

    raise error


def split_doc_into_chunks(doc_content: str, lang: str, chunk_size: int):
    if lang == 'zh':
        token_per_byte = 0.248
    else:
        assert lang == 'en'
        token_per_byte = 0.263
        
    def token_count(doc_content):
        content_bytes = len(doc_content.encode())
        return math.ceil(content_bytes * token_per_byte)
    
    doc_token_cnt = token_count(doc_content)
    chunk_nums = math.ceil(doc_token_cnt / chunk_size) 
    char_chunk_size = math.ceil(len(doc_content) / chunk_nums)

    chunks = []
    for i in range(chunk_nums):
        cur_chunk = doc_content[i*char_chunk_size: (i+1)*char_chunk_size]
        chunks.append(cur_chunk)
    return chunks


def extract_questions(doc_content: str, lang: str, model_name: str, api_base: str, chunk_size: int, prompt: dict):
    chunks = split_doc_into_chunks(doc_content, lang, chunk_size)
    qe_prompt = prompt[lang]

    doc_questions = []
    for cur_chunk in chunks:
        if len(cur_chunk) == 0:
            continue

        questions = get_llm_response(qe_prompt.format(chunk=cur_chunk), model_name, api_base)
        questions = questions.strip()
        questions = re.findall('"([^"]*)"', questions)
        questions = [q for q in questions if q.endswith('？') or q.endswith('?')]
        questions = list(set(questions))

        doc_questions.append(questions)
    
    return doc_questions, chunks


def filter_questions(questions: list, lang: str, model_name: str, api_base: str, prompt: dict):
    fq_prompt = prompt[lang]
    
    total_filtered_labels = []
    for indoc_qs in questions:
        indoc_filtered_labels = []
        for qs in indoc_qs:
            chunk_filtered_labels = []
            for q in qs:
                fscore = get_llm_response(fq_prompt.format(question=q), model_name, api_base)
                if '1' in fscore:
                    chunk_filtered_labels.append(True)
                else:
                    chunk_filtered_labels.append(False)
            indoc_filtered_labels.append(chunk_filtered_labels)

        total_filtered_labels.append(indoc_filtered_labels)
    
    return total_filtered_labels


def generate_anwer(chunk: str, question: str, lang: str, model_name: str, api_base: str, prompt: dict):
    ag_prompt = prompt[lang]
        
    answer = get_llm_response(ag_prompt.format(chunk=chunk, question=question), model_name, api_base)
    
    return answer


def simplify_qa(qa_pairs: List[dict], lang: str, model_name: str, api_base: str, prompt: dict):
    qk, ak = 'question', 'answer'
    sqa_prompt = prompt[lang]

    filtered_qa_pairs = []
    for qi in range(0, len(qa_pairs), 5):
        cur_qa_pairs = qa_pairs[qi:qi+5]
        for qa in cur_qa_pairs:
            if type(qa) is str:
                qa = json.loads(qa)
                
            a = qa[ak]
            if len(a) == 0:
                break
            filtered_qa_pairs.append(qa)

    simplified_qa_pairs = []
    for qa in filtered_qa_pairs:
        q, a = qa[qk], qa[ak]
        a = a.strip('答案：')
        sim_a = get_llm_response(sqa_prompt.format(q=q, a=a), model_name, api_base)
        simplified_qa_pairs.append({qk:q, ak:sim_a})
    
    return simplified_qa_pairs


def rewrite_answer(qa_pair: dict, lang: str, model_name: str, api_base: str, prompt: dict):
    ra_prompt = prompt[lang]
    qk, ak = 'question', 'answer'

    q = qa_pair[qk]
    a = qa_pair[ak]

    format_answer = get_llm_response(ra_prompt.format(q=q, a=a), model_name, api_base)

    return format_answer


def merge_qa_couples(qa1: str, qa2: str, lang: str, model_name: str, api_base: str, prompt: dict):
    if 'question' in qa2 and lang == 'zh':
        qa2 = {
            '问题': qa2['question'],
            '回答': qa2['answer']
        }
    complex_qa_generation = prompt[lang]
    cross_doc_qa = get_llm_response(complex_qa_generation.format(qa1=qa1, qa2=qa2), model_name, api_base)
    return cross_doc_qa



if __name__ == '__main__':
    model_name = 'Qwen2-72B-Instruct' # gpt-4
    api_base = 'http://10.1.22.125:8000/v1'

    doc1 = ""
    doc2 = "偶然遇到一个跳广场舞的大妈跟她的女儿抱怨，说跳舞时经常会出现漏尿的情况，然后她周边的好几个阿姨也是这样。\n所以大妈很沮丧，认为这是年纪大了的缘故，年纪大所以身体出问题了，不想服老都不行。\n其实像大妈这种情况的有很多，大家也可以发现做盆地康复的都是产后的新晋宝妈们，很少见到中年和老年妇女的身影，那这代表中老年妇女没有盆底的问题吗？\n显然不是。其实这类人群的盆底问题才是最严重，亟需进行干预的。\n其中一部分人是产后没有进行及时的康复遗留的问题。\n因为问题部位比较隐私，所以也就忍忍过去了，长时间和尿不湿为伴。\n还有一部分集中在绝经后妇女身上。除了生产会造成盆底的损伤，衰老和雌激素减少也是造成盆底功能障碍的原因。我们的膀胱也会衰老，衰老后的膀胱导致膀胱颈及近端尿道下移，尿道松弛变短。\n雌激素是保持盆底组织张力和弹性的重要因素，雌激素下降后，盆底肌肌张力便也会伴随下降，对于盆腔脏器的承托力也下降，从而产生压力性尿失禁、子宫脱垂、盆腔痛等症状。\n尽管现在随着网络的发展，很多医学科普得以被广泛且迅速的传播，产后康复的重要性被很多人熟知。\n但是不乏因为经济时间等原因错过了产后的最佳康复时机，或者我们的母辈或者奶奶辈由于当时的医疗认知和条件受限，一直都未接受产后康复，长期伴随漏尿等人群。\n产后康复最佳时机为产后42天至6个月，那么有很多人就有疑问了：\n如果我产后3个月内没有进行及时的康复或者已经当妈妈好多年，还有必要进行康复吗？康复的效果会如何？\n在解释这个问题之前，我们理解什么叫做黄金期：\n● 黄金期：\n任何治疗的黄金期都是指在此阶段介入治疗比其他任何时候，其恢复质量和速度都是最佳的，因为此时盆底功能只是检测到出现问题，还没有伴随频繁的漏尿和严重的脱垂产生，此外还有自身神经和细胞修复都处于一个高峰期。\n但这并不代表着其他阶段介入治疗便无效，只是康复速度相对来讲可能会受到影响。\n再次建议如果有条件还是花时间和精力尽早开始产后康复，如果真的错过了产后恢复黄金期，或者已经四五十岁的妈妈辈甚至五六十岁的奶奶辈：\n如果尚存咳嗽漏尿、盆腔痛、小腹下坠感等盆底功能障碍症状，也请不要讳疾忌医，只要介入康复 ，就有恢复的可能性。盆底肌功能靠自愈是不太可能恢复正常的，还是要进行专业的盆底康复训练。所以盆底恢复不只限于产后6个月内做，凡是具有盆底肌功能障碍的都有必要进行盆底康复。\n作者介绍\n毕霞\n毕霞，医学博士，主任医师，硕士研究生导师，现任上海健康医学院附属周浦医院康复医学科主任、上海健康医学院康复学院副院长。2010年、2018年、2019年先后入选浦东新区卫生系统优秀学科带头人和领先人才培养计划。以第一作者和通讯作者发表SCI论文4篇，核心期刊论文30余篇。\n专业方向：脑卒中康复、骨折术后康复、脊髓损伤康复、人工关节围手术期康复、手外伤康复等。\n主要学术任职：中国康复医学会老年康复分会委员、中国康复医学会医养结合专委会委员、中国医师协会老年康复分会康复委员、上海市康复医学会常务理事，上海市医学会物理医学与康复学分会委员，上海市康复医学会康复治疗委员会副主任委员，上海市康复医学会骨科康复委员会委员，上海市康复医学会脊柱脊髓损伤康复专委会委员，上海市康复医学会社区康复专委会委员，上海市康复医学会体育保健康复专委会委员、上海市浦东新区医学会康复理疗专委会主任委员等学术任职。\n上海市科委科普项目资助\n（项目编号：20DZ2311100）"


    doc3 = "Frederick Newmeyer\n\nFrederick J. (Fritz) Newmeyer (born January 30, 1944) is Professor Emeritus of Linguistics at the University of Washington and adjunct professor in the University of British Columbia Department of Linguistics and the Simon Fraser University Department of Linguistics. He has published widely in theoretical and English syntax and is best known for his work on the history of generative syntax and for his arguments that linguistic formalism (i.e. generative grammar) and linguistic functionalism are not incompatible, but rather complementary. In the early 1990s he was one of the linguists who helped to renew interest in the evolutionary origin of language. More recently, Newmeyer argued that facts about linguistic typology are better explained by parsing constraints than by the principles and parameters model of grammar. Nevertheless, he has continued to defend the basic principles of generative grammar, arguing that Ferdinand de Saussure's langue/parole distinction as well Noam Chomsky's distinction between linguistic competence and linguistic performance are essentially correct.\n\nBiography\nNewmeyer was born in Philadelphia, but grew up in Port Washington, New York. He received his BA in geology from the University of Rochester in 1965 and his MA in linguistics from that same institution two years later. Newmeyer was awarded a PhD in linguistics from the University of Illinois in 1969, writing a dissertation entitled English Aspectual Verbs under the direction of Robert B. Lees. His only permanent position has been in the Department of Linguistics at the University of Washington (from 1969 until his retirement in 2006), but he has held visiting positions at a variety of universities around the world, including the University of Edinburgh, Wayne State University, University of London, Cornell University, University of Maryland, UCLA, La Trobe University, Universidade de São Paulo, Universidad Nacional del Comahue, Universiteit van Tilburg, Heinrich-Heine-Universität Düsseldorf, École Normale Supérieure, Institut des Science Cognitives, Max Planck Institute for Evolutionary Anthropology, and University of Ljubljana. In 2002, Newmeyer was President of the Linguistic Society of America, from 2003-2006 Howard and Frances Nostrand Professor of Linguistics at Washington, and in 2006 he was elected Fellow of the American Association for the Advancement of Science and the Linguistic Society of America. In his 20s and 30s Newmeyer was heavily involved in left politics, being an active member of Students for a Democratic Society in the late 1960s and of the International Socialists from 1971 to 1977. He was married to Carolyn Platt between 1968 and 1973 and in 1993 he married Marilyn Goebel, who managed the internal web pages for Group Health Cooperative in Seattle before her retirement in 2003. In 2006, he and Goebel moved to Vancouver.\n\nPublications\n\nBooks written\n2005. Possible and Probable Languages: A Generative Perspective on Linguistic Typology. Oxford: Oxford University Press.\n1998. Language Form and Language Function.  Cambridge, MA:  MIT Press.\n1996. Generative Linguistics: A Historical Perspective.  London: Routledge.\n1986. The Politics of Linguistics. Chicago: University of Chicago Press. Japanese translation 1994, Tokyo: Iwanami Shoten Publishers. Arabic translation 1997, Abha (Saudi Arabia): The Literary Club. Persian translation 2002, Ney (Iran).\n1983. Grammatical Theory: Its Limits and Its Possibilities. Chicago: University of Chicago Press. Malay translation 1996, Kuala Lumpur: Dewan Bahasa dan Pustaka.\n1980. Linguistic Theory in America: The First Quarter Century of Transformational Generative Grammar. New York: Academic Press. Second edition 1986. First edition translated into Spanish  1982, Madrid: Alianza Editorial. Translations of second edition: Korean 1995, Seoul: Kul Press. Chinese 1998, Taipei: Crane Publishing Co, Ltd.; Japanese translation under contract.\n1975. English Aspectual Verbs. The Hague: Mouton and Company.\n\nBooks edited\n1998. Functionalism and Formalism in Linguistics (with Michael Darnell, Edith Moravcsik, Michael Noonan, and Kathleen Wheatley). Studies in Language Companion Series, Volume 41. Amsterdam: John Benjamins.\nVolume I: General Papers.\nVolume II: Case Studies.\n1988. Linguistics: The Cambridge Survey. Cambridge: Cambridge University Press. Spanish translation published 1990-1992 as Panorama de la Lingüística Moderna, Madrid:Visor Distribuciones.\nVolume I: Linguistic Theory: Foundations.\nVolume II: Linguistic Theory: Extensions and Implications.\nVolume III: Language: Psychological and Biological Aspects.\nVolume IV: Language: The Socio-Cultural Context.\n1986. A Festschrift for Sol Saporta (with Michael Brame and Heles Contreras).  Linguistic Research Monograph Series Publication. Seattle: Noit Amrofer Press.\n\nReferences\n\nExternal links\n\nHomepage\n\nCategory:1944 births\nCategory:Living people\nCategory:Linguists from the United States\nCategory:Generative linguistics\nCategory:Syntacticians\nCategory:University of Washington faculty\nCategory:Wayne State University faculty\nCategory:University of British Columbia faculty\nCategory:Simon Fraser University faculty\nCategory:People from Port Washington, New York\nCategory:Fellows of the American Association for the Advancement of Science\nCategory:Fellows of the Linguistic Society of America\nCategory:Linguistic Society of America presidents\n"
    doc4 = "The 6' Choo Choo Rug from Flagship Carpets features a colorful train of baby animals. It's constructed of dense nylon fiber that's treated with Scotchguard stain protection. The edges are tightly bound and double-stitched for maximum strength. A permanent antimicrobial treatment protects against odors, mold and mildew. And the skid-resistant Action Bac woven backing system prevents wrinkles and bunching. Printed with Flagship's high-tech process, the Choo Choo Rug boasts crisp, defined images and fade-resistant saturated color. Made in the USA and backed by a lifetime abrasive wear warranty."
    lang = 'en'

    # extracted_qa_pairs1 = get_in_doc_qa_pairs(doc1, lang, model_name, api_base)
    # print(print('>>> 文档1中的所有问题：'))
    # print(extracted_qa_pairs1)

    # extracted_qa_pairs2 = get_in_doc_qa_pairs(doc2, lang, model_name, api_base)
    # print(print('>>> 文档2中的所有问题：'))
    # print(extracted_qa_pairs2)

    # ridx1 = random.randint(0, len(extracted_qa_pairs1)-1)
    # ridx2 = random.randint(0, len(extracted_qa_pairs2)-1)
    # qa1 = extracted_qa_pairs1[ridx1]
    # print('>>> 文档1中的随机问题：')
    # print(f'>>> {qa1}')
    # qa2 = extracted_qa_pairs2[ridx2]
    # print('>>> 文档2中的随机问题：')
    # print(f'>>> {qa2}')
    # print('>>> 两个文档结合的复杂问题：')
    # complex_qa = merge_qa_couples(qa1, qa2, lang, model_name)
    # print(complex_qa)

    # extracted_sums1 = get_in_doc_sum(doc3, lang, model_name, api_base)

    # extracted_sums2 = get_in_doc_sum(doc4, lang, model_name, api_base)

    # print('>>> 文档1中摘要：')
    # print(f'>>> {extracted_sums1[0]}')
    # print('>>> 文档2中摘要：')
    # print(f'>>> {extracted_sums2[0]}')
    # print('>>> 融合文档摘要：')
    # merged_sum = merge_sum_couples(extracted_sums1[0], extracted_sums2[0], lang, model_name, api_base)
    
    # chunk_size = 2048
    # questions = extract_wiki_questions(doc1, lang, model_name, api_base, chunk_size)
    # print(questions)

    # questions = [["When was Frederick J. Newmeyer born?", "What is Frederick J. Newmeyer's academic position at the University of Washington?", "What are some of the universities where Frederick J. Newmeyer has held visiting positions?", "What is Frederick J. Newmeyer known for in the field of linguistics?", "In which decade did Frederick J. Newmeyer help renew interest in the evolutionary origin of language?", "What argument has Frederick J. Newmeyer made regarding linguistic typology?", "What basic principles of generative grammar has Frederick J. Newmeyer continued to defend?", "Where did Frederick J. Newmeyer grow up?", "What degrees did Frederick J. Newmeyer receive from the University of Rochester?", "What was the title of Frederick J. Newmeyer's PhD dissertation?", "Where did Frederick J. Newmeyer hold his only permanent academic position?", "In what year did Frederick J. Newmeyer retire from the University of Washington?", "In what year was Frederick J. Newmeyer President of the Linguistic Society of America?", "What professorship did Frederick J. Newmeyer hold at the University of Washington from 2003-2006?", "In what year was Frederick J. Newmeyer elected Fellow of the American Association for the Advancement of Science and the Linguistic Society of America?", "What was Frederick J. Newmeyer's involvement in politics in his 20s and 30s?", "To whom was Frederick J. Newmeyer married in 1968 and 1993?", "In what year did Frederick J. Newmeyer and Marilyn Goebel move to Vancouver?", "What are some of the books written by Frederick J. Newmeyer?", "What are some of the books edited by Frederick J. Newmeyer?"]]
    # questions = [[]]
    # qa_pairs = generate_wiki_anwers(doc1, questions, lang, model_name, api_base, chunk_size)
    # print(qa_pairs)

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

    qa_pairs = ["{\"question\": \"What is the name of Chris Rea's second studio album?\", \"answer\": \"Deltics\"}", "{\"question\": \"In what year was Deltics released?\", \"answer\": \"1979\"}", "{\"question\": \"Which record label released the album Deltics?\", \"answer\": \"Magnet Records\"}", "{\"question\": \"What is the album Deltics named after?\", \"answer\": \"East Coast rail network's Deltic-class locomotives\"}", "{\"question\": \"What is the highest position Deltics reached on the UK Albums Chart?\", \"answer\": \"number fifty-four\"}", "{\"question\": \"Which single from the album Deltics charted on both the UK Singles Chart and Billboard Hot 100?\", \"answer\": \"The single 'Diamonds' charted on both the UK Singles Chart and Billboard Hot 100.\"}", "{\"question\": \"How long did the single 'Diamonds' chart on the Billboard Hot 100?\", \"answer\": \"The single 'Diamonds' charted on the Billboard Hot 100 for eight weeks.\"}", "{\"question\": \"What is the B-side of the single 'Diamonds'?\", \"answer\": \"The B-side of the single 'Diamonds' is 'Cleveland Calling'.\"}", "{\"question\": \"Is the B-side of the single 'Diamonds' included on the CD reissue of the album?\", \"answer\": \"No, the B-side of the single 'Diamonds', 'Cleveland Calling', is not included on the CD reissue of the album.\"}", "{\"question\": \"Who produced the album Deltics?\", \"answer\": \"The album Deltics was produced by Gus Dudgeon.\"}", "{\"question\": \"Which song on the album Deltics is the longest?\", \"answer\": \"The longest song on the album Deltics is \\\"Cenotaph/Letter from Amsterdam\\\" which is 5:49 long.\"}", "{\"question\": \"How many songs are on the album Deltics?\", \"answer\": \"There are 11 songs on the album Deltics.\"}", "{\"question\": \"Which song on the album Deltics is the shortest?\", \"answer\": \"The shortest song on the album Deltics is \\\"No Qualifications\\\" which is 2:20 long.\"}", "{\"question\": \"Who played drums on the album Deltics?\", \"answer\": \"Drums on the album Deltics were played by Dave Mattacks and Adrian Rea.\"}", "{\"question\": \"Which two songs were released as singles from the album Deltics?\", \"answer\": \"The two songs released as singles from the album Deltics are \\\"Diamonds\\\" and \\\"Raincoat and a Rose\\\".\"}", "{\"question\": \"What category does the album Deltics fall under in terms of production?\", \"answer\": \"Albums produced by Gus Dudgeon\"}", "{\"question\": \"What category does the album Deltics fall under in terms of record label?\", \"answer\": \"Magnet Records albums\"}"]
    sqa_pairs = simplify_qa(qa_pairs, lang, model_name, api_base, simplify_qa_prompt)
    print(sqa_pairs)