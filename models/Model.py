import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class BertForMultilabelSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert.config.output_hidden_states = True
        
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
                return_dict=None):
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
        
        """
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        """
        
        all_hidden_states = torch.stack(outputs[2])
        concatenate_pooling = torch.cat(
            (torch.squeeze(torch.index_select(all_hidden_states[-1], 1, torch.tensor([0]).to('cuda'))),
             torch.squeeze(torch.index_select(all_hidden_states[-2], 1, torch.tensor([0]).to('cuda'))),
             outputs[1]
             ),
            -1)
        logits = self.linear(concatenate_pooling)
        logits = self.dropout(logits)
        logits = self.classifier(logits)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.float().view(-1, self.num_labels))
        
        return loss, logits
