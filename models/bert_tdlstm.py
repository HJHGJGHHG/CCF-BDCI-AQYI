import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from models.layers import DynamicLSTM


def get_x_l(character_in_text, all_hidden_states):
    x_l_sum = torch.zeros(1, 64, 1024).to('cuda')
    x_l_len = []
    for i_sample in range(8):  # batch_size
        len_l = character_in_text[i_sample].tolist()[0] + 1
        x_l_len.append(len_l)
        x_l = torch.index_select(all_hidden_states[-1][i_sample], 0,
                                 torch.tensor(list(range(1, len_l + 1))).to('cuda'))
        zero_l = torch.zeros(64 - len_l, 1024).to('cuda')
        x_l = torch.unsqueeze(torch.cat((x_l, zero_l), dim=0), dim=0)
        x_l_sum = torch.cat((x_l_sum, x_l), dim=0)
    x_l_len = torch.LongTensor(x_l_len).to('cuda')
    x_l_sum = torch.index_select(x_l_sum, 0, torch.tensor(list(range(1, args.batch_size + 1))).to('cuda'))
    return x_l_sum, x_l_len


class Bert_TD_LSTM(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert.config.output_hidden_states = True
        
        self.lstm_l = DynamicLSTM(1024, 1024, num_layers=1, batch_first=True)
        self.lstm_r = DynamicLSTM(1024, 1024, num_layers=1, batch_first=True)
        
        self.linear = nn.Sequential(
            nn.Linear(3 * 1024, 1024),
            nn.Tanh())
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                character_in_text=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)
        
        all_hidden_states = outputs[2]
