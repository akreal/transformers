import sentencepiece as spm
import torch

class OWSMEmbeddingsFeatureExtractor:
    def __init__(self):
        self.sp = None
        self.ctc_weight = None
        self.ctc_bias = None

    def _build(self):
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load("owsm_ctc_v3.1_1B/data/token_list/bpe_unigram50000/bpe.model")
            params = torch.load("owsm_ctc_v3.1_1B/exp/s2t_train_s2t_multitask-ctc_ebf27_conv2d8_size1024_raw_bpe50000/valid.total_count.ave_5best.till45epoch.pth")
            self.ctc_weight = params["ctc.ctc_lo.weight"]
            self.ctc_bias = params["ctc.ctc_lo.bias"]

    def __call__(self, inputs):
        self._build()
        tokens = [t-1 for t in self.sp.EncodeAsPieces(inputs)]
        return torch.stack([self.ctc_weight[t] + self.ctc_bias[t] for t in tokens])
