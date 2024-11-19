import os
import json

from datasets.features import Image as ImageFeature
from datasets import load_dataset, concatenate_datasets, DatasetDict

from src.logger import logger
from src.entity.entity import DataIngestorConfig

class DataIngestor:
    def __init__(self, config:DataIngestorConfig):
        self.config=config

    def create_dataset(self, train_size=0.9):
        columns=['image','label_bbox_enriched']
        dataset = load_dataset(self.config.DATASET_NAME).select_columns(columns).cast_column('image', ImageFeature(mode='RGB'))
        dataset=concatenate_datasets([dataset['train'], dataset['test']])

        tmp=dataset.train_test_split(train_size=train_size)
        self.dataset=DatasetDict()
        self.dataset['train']=tmp.pop('train')
        tmp=tmp['test'].train_test_split(train_size=0.5)
        self.dataset['validation']=tmp.pop('train')
        self.dataset['test']=tmp.pop('test')

        self.dataset=self.dataset.map(self.preprocess, batched=True, batch_size=1024, remove_columns=['label_bbox_enriched'])
        self.dataset.save_to_disk(self.config.ARTIFACT_DIR)

        return self.dataset

    @staticmethod
    def preprocess(batch):
        target=[x if x is not None else [] for x in batch['label_bbox_enriched']]
        return {'label_bbox': target}

    def create_label_mapping(self):
        labels={}
        for split in ['train', 'validation', 'test']:
            for sample in self.dataset[split]['label_bbox']:
                for box in sample:
                    label=box['label']
                    if label in labels.keys():
                        labels[label]+=1
                    else:
                        labels[label]=1

        labels=sorted(labels.items(), key=lambda x: -x[1])
        labels={x[0]:x[1] for x in labels}
        label_to_id = {label:id for id, label in enumerate(labels.keys())}
        id_to_label = {id:label for id,label in enumerate(labels.keys())}
        
        with open(self.config.ARTIFACT_DIR+'/label_count.json', 'w') as file:
            json.dump(labels, file, indent=4)

        with open(self.config.ARTIFACT_DIR+'/label_to_id.json', 'w') as file:
            json.dump(label_to_id, file, indent=4)

        with open(self.config.ARTIFACT_DIR+'/id_to_label.json', 'w') as file:
            json.dump(id_to_label, file, indent=4)

    def sync_to_s3(self):
        command = f'aws s3 sync {self.config.ARTIFACT_DIR} s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR}'
        os.system(command)
        logger.info(f"Sync data from {self.config.ARTIFACT_DIR} to s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR}")
    
    def sync_from_s3(self):
        command = f'aws s3 sync s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR} {self.config.ARTIFACT_DIR}'
        os.system(command)
        logger.info(f"Sync data from s3://{self.config.BUCKET_NAME}/{self.config.CLOUD_DIR} to {self.config.ARTIFACT_DIR}")