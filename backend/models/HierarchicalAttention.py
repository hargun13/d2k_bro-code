import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalAttention(nn.Module):
    def __init__(self, num_classes, sequence_length, num_sentences, vocab_size, embed_size, hidden_size):
        super(HierarchicalAttention, self).__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Word Encoder
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru_word = nn.GRU(embed_size, hidden_size, bidirectional=True, batch_first=True)

        # Word Attention
        self.W_w_attention_word = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.context_vector_word = nn.Parameter(torch.Tensor(hidden_size * 2))

        # Sentence Encoder
        self.gru_sentence = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)

        # Sentence Attention
        self.W_w_attention_sentence = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.context_vector_sentence = nn.Parameter(torch.Tensor(hidden_size * 2))

        # Fully Connected layer + Softmax
        self.fc = nn.Linear(hidden_size * 4, num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.context_vector_word)
        nn.init.normal_(self.context_vector_sentence)

    def attention_word_level(self, hidden_state):
        hidden_state = torch.stack(hidden_state, dim=1)
        hidden_state_2 = hidden_state.view(-1, self.hidden_size * 2)
        hidden_representation = torch.tanh(self.W_w_attention_word(hidden_state_2))
        attention_logits = torch.sum(hidden_representation * self.context_vector_word, dim=1)
        attention_logits_max = torch.max(attention_logits, dim=1, keepdim=True)[0]
        p_attention = F.softmax(attention_logits - attention_logits_max, dim=1)
        p_attention_expanded = p_attention.unsqueeze(2)
        sentence_representation = torch.sum(p_attention_expanded * hidden_state, dim=1)
        return sentence_representation

    def attention_sentence_level(self, hidden_state_sentence):
        hidden_state = torch.stack(hidden_state_sentence, dim=1)
        hidden_representation = torch.tanh(self.W_w_attention_sentence(hidden_state.view(-1, self.hidden_size * 4)))
        attention_logits = torch.sum(hidden_representation * self.context_vector_sentence, dim=1)
        attention_logits_max = torch.max(attention_logits, dim=1, keepdim=True)[0]
        p_attention = F.softmax(attention_logits - attention_logits_max, dim=1)
        p_attention_expanded = p_attention.unsqueeze(2)
        sentence_representation = torch.sum(p_attention_expanded * hidden_state, dim=1)
        return sentence_representation

    def forward(self, input_x):
        embedded_words = self.embedding(input_x)
        embedded_words_reshaped = embedded_words.view(-1, self.sequence_length, self.embed_size)

        hidden_state_word, _ = self.gru_word(embedded_words_reshaped)

        sentence_representation = self.attention_word_level(hidden_state_word)
        sentence_representation_reshaped = sentence_representation.view(-1, self.num_sentences, self.hidden_size * 2)

        hidden_state_sentence, _ = self.gru_sentence(sentence_representation_reshaped)

        document_representation = self.attention_sentence_level(hidden_state_sentence)

        logits = self.fc(document_representation)
        return logits
