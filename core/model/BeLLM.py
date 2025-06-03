# -*- coding: utf-8 -*-
import torch
from angle_emb import *
from transformers import PreTrainedTokenizer
import copy


class MyAnglE(AngleBase):
    cfg_file_name = 'angle.config'
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 pooling_strategy: Optional[str] = None,
                 train_mode: bool = True,
                 device: Optional[str] = None,
                 tokenizer_padding_side: Optional[str] = None,
                 **kwargs: Any):
        super().__init__()
        self.max_length = 1024
        self.train_mode = train_mode
        self.pooling_strategy = pooling_strategy
        self.is_llm = True
        if device:
            self.device = device
        else:
            self.device = set_device()


        self.gpu_count = 1

        self.tokenizer = tokenizer
        if tokenizer_padding_side is not None and self.tokenizer.padding_side != tokenizer_padding_side:
            self.tokenizer.padding_side = tokenizer_padding_side
        if self.is_llm and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0


        self.backbone = model
        self.backbone.config.use_cache = False
        self.pooler = Pooler(
            self.backbone,
            pooling_strategy=self.pooling_strategy,
            padding_side=self.tokenizer.padding_side)
    def cuda(self):
        if self.gpu_count > 1 and self.is_llm:
            return self
        self.backbone = self.backbone.to(torch.device(self.device))
        return self

    def to(self, device: Any):
        if isinstance(device, str):
            device = torch.device(device)
        self.backbone = self.backbone.to(device)
        self.device = device
        return self

    def fit(self,
            train_ds: Dataset,
            valid_ds: Optional[Dataset] = None,
            args:Optional[TrainingArguments]=None,
            loss_kwargs: Optional[Dict] = None,
            apply_ese: bool = False,
            filter_duplicate: bool = True,
            push_to_hub: bool = False,
            hub_model_id: Optional[str] = None,
            hub_private_repo: bool = True,
            coword_random_mask_rate: float = 0.,
            padding: str = 'longest'):

        # save config
        #self.save_config(os.path.join(output_dir, AnglE.cfg_file_name))
        # save tokenizer
        #self.tokenizer.save_pretrained(output_dir)

        if self.gpu_count > 1:
            args.gradient_accumulation_steps = args.gradient_accumulation_steps // self.gpu_count
        callbacks = None


        CustomTrainer = AngleESETrainer if apply_ese else AngleTrainer
        trainer = CustomTrainer(
            pooler=self.pooler,
            model=self.backbone,
            dataset_format=self.detect_dataset_format(train_ds),
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            loss_kwargs=loss_kwargs,
            tokenizer=self.tokenizer,
            args=args,
            callbacks=callbacks,
            data_collator=AngleDataCollator(
                self.tokenizer,
                padding=padding,
                return_tensors="pt",
                max_length=self.max_length,
                filter_duplicate=filter_duplicate,
                coword_random_mask_rate=coword_random_mask_rate,
            )
        )
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.backbone = torch.compile(self.backbone)
        trainer.train()
        #self.backbone.save_pretrained(args.output_dir)
    def evaluate(self, data: Dataset, batch_size: int = 32, metric: str = 'spearman_cosine') -> float:
        """ evaluate

        :param data: Dataset, DatasetFormats.A is required
        :param batch_size: int. Default 32.
        :param metric: str. Default 'spearman_cosine'.

        :return: float.
        """
        return CorrelationEvaluator(
            text1=data['text1'],
            text2=data['text2'],
            labels=data['label'],
            batch_size=batch_size,
        )(self)[metric]

    def truncate_layer(self, layer_index: int):
        """ truncate layer

        :param layer_index: int. layers after layer_index will be truncated.
        :return: self
        """
        if len(self.backbone.encoder.layer) < layer_index:
            logger.info('current layer_index is larger than the number of layers, please check whether it is correct')
        self.backbone.encoder.layer = self.backbone.encoder.layer[:layer_index]
        return self

    def encode(self,
               inputs: Union[List[str], Tuple[str], List[Dict], str],
               max_length: Optional[int] = None,
               to_numpy: bool = True,
               embedding_start: int = 0,
               embedding_size: Optional[int] = None,
               device: Optional[Any] = None,
               prompt: Optional[str] = None,
               normalize_embedding: bool = False,
               padding: str = 'longest'):
        """
        encode texts.

        :param inputs: Union[List[str], Tuple[str], List[Dict], str]. Input texts. Required.
        :param max_length: Optional[int]. Default None.
        :param to_numpy: bool. Default True.
        :param embedding_start: int. Specify the start position of the embedding (for Espresso).
        :param embedding_size: Optional[int]. Specify embedding size (for Espresso).
            The embeddings from embedding_start to embedding_start+embedding_size will be returned.
        :param device: Optional[Any]. Default None.
        :param prompt: Optional[str]. Default None.
        :param normalize_embedding: bool. Default False.
        :param padding: str. Padding strategy of tokenizer. Default 'longest'.
        """
        self.backbone.eval()

        if device is None:
            device = self.device
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
        if prompt is not None:
            for i, obj in enumerate(inputs):
                assert isinstance(obj, dict), 'The prompt has been set, please pass a dict like {"prompt_key": "text"}'
                inputs[i] = prompt.format(**obj)

        tok = self.tokenizer(
            inputs,
            padding=padding,
            max_length=max_length or self.max_length,
            truncation=True,
            return_tensors='pt')
        tok.to(device)
        with torch.no_grad():
            output = self.pooler(tok,
                                 embedding_start=embedding_start,
                                 embedding_size=embedding_size)
        if normalize_embedding:
            output = nn.functional.normalize(output, p=2, dim=-1)
        if to_numpy:
            return output.float().detach().cpu().numpy()
        return output

    def push_to_hub(self, hub_model_id: str, private: bool = True, exist_ok: bool = False, **kwargs):
        """ push model to hub

        :param hub_model_id: str, hub model id.
        :param private: bool, whether push to private repo. Default True.
        :param exist_ok: bool, whether allow overwrite. Default False.
        :param kwargs: other kwargs for `push_to_hub` method.
        """
        if not exist_ok and repo_exists(hub_model_id):
            raise ValueError(f"Model {hub_model_id} already exists on the hub. Set `exist_ok=True` to overwrite.")
        self.tokenizer.push_to_hub(hub_model_id, private=private, **kwargs)
        self.backbone.push_to_hub(hub_model_id, private=private, **kwargs)

    def save_pretrained(self, output_dir: str, exist_ok: bool = True):
        """ save model and tokenizer

        :param output_dir: str, output dir.
        :param exist_ok: bool, whether allow overwrite. Default True.
        """
        if not exist_ok and os.path.exists(output_dir):
            raise ValueError(f"Output directory ({output_dir}) already exists and is not empty.")
        os.makedirs(output_dir, exist_ok=exist_ok)
        self.tokenizer.save_pretrained(output_dir)
        self.backbone.save_pretrained(output_dir)
    def detect_dataset_format(self, ds: Dataset):
        for obj in ds:
            return obj['extra']['dataset_format']