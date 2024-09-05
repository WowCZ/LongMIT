'''
Description: 
Version: 1.0
Autor: Zhi Chen
Date: 2024-07-17 12:03:53
LastEditors: Zhi Chen
LastEditTime: 2024-07-17 12:18:29
'''

from abc import ABC, abstractmethod
from omegaconf import OmegaConf


class AgentFlow(ABC):
    def __init__(self, config_file: str) -> None:
        self.cfg = OmegaConf.load(config_file)

    @abstractmethod
    def run(self):
        '''
        description: closed high-quality long-qa extraction pipeline with efficient and active human-in-loop
        return {*}
        '''        
        pass