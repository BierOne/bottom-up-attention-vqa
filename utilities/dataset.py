import os, json
import h5py
import torch
import torch.utils.data as data
from utilities import config, utils


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word2vocab(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        json.dump([self.word2idx, self.idx2word], open(path, 'w'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = json.load(open(path, 'r'))
        d = cls(word2idx, idx2word)
        return d

    def add_word2vocab(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def get_loader(split):
    """ Returns a data loader for the desired split """
    assert split in ['train', 'val','trainval', 'test']
    image_feature_path = config.rcnn_trainval_path if split != 'test' else config.rcnn_test_path
    dataset = VQAFeatureDataset(
        split,
        image_feature_path,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=512,
        shuffle=True if split not in ['val', 'test'] else False,  # only shuffle the data in training
        pin_memory=True,
        num_workers=config.workers,
    )
    return loader


class VQAFeatureDataset(data.Dataset):
    def __init__(self, split, image_features_path, dataroot=config.cache_root):
        super(VQAFeatureDataset, self).__init__()
        # assert split in ['train', 'val','trainval', 'test']
        self.split = split
        self.dataroot = dataroot
        self.image_features_path = image_features_path
        self.label2ans = json.load(open(os.path.join(dataroot, 'trainval_label2ans.json'), 'r'))
        self.dictionary = Dictionary.load_from_file(os.path.join(dataroot, 'dictionary.json'))

        self.num_ans_candidates = len(self.label2ans)
        self.img_id2idx = self._create_img_id_to_idx()
        self.entries = self._load_entries()

    def _create_img_id_to_idx(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _load_entries(self):
        """Load entries
        img_id2idx: dict {img_id -> idx} val can be used to retrieve image or features
        dataroot: root path of dataset
        split: 'train', 'val', 'trainval', 'test'
        """
        def _create_entry(img_id, question, answer):
            entry = {
                'question_id': question['question_id'],
                'image_id': img_id,
                'img_idx': self.img_id2idx[img_id],
                'question': self.encode_question(question['question']),
                'answer': self.encode_answer(answer)
            }
            return entry

        questions = utils.get_file(self.split, question=True)
        questions = sorted(questions, key=lambda x: x['question_id'])
        if self.split != 'test':
            answer_path = os.path.join(self.dataroot, '%s_target.json' % self.split)
            with open(answer_path, 'r') as fd:
                answers = json.load(fd)
            utils.assert_eq(len(questions), len(answers))
            answers = sorted(answers, key=lambda x: x['question_id'])
            entries = []
            for question, answer in zip(questions, answers):
                img_id = answer.pop('image_id')
                ques_id = answer.pop('question_id')
                utils.assert_eq(question['question_id'], ques_id)
                utils.assert_eq(question['image_id'], img_id)
                entries.append(_create_entry(img_id, question, answer))
        else:
            entries = [_create_entry(question['image_id'], question, 0) for question in questions]
        print(len(entries))
        return entries

    def load_image(self, img_idx):
        # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
        # forks for multiple works, every child would use the same file object and fail.
        # Having multiple readers using different file objects is fine though, so we just init in here.
        if not hasattr(self, 'features_file'):
            self.features_file = h5py.File(self.image_features_path, 'r')
        feature = self.features_file.get('features')[img_idx]
        spatials = self.features_file.get('boxes')[img_idx]
        return torch.from_numpy(feature), torch.from_numpy(spatials)

    def encode_question(self, question, max_length=config.max_question_len):
        tokens = self.dictionary.tokenize(question, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
            tokens = padding + tokens
        utils.assert_eq(len(tokens), max_length)
        return torch.LongTensor(tokens)

    def encode_answer(self, answer):
        if self.split == 'test':
            return torch.zeros(0)
        target = torch.zeros(self.num_ans_candidates)
        labels = torch.LongTensor(answer['labels'])
        scores = torch.FloatTensor(answer['scores'])
        if len(labels):
            target.scatter_(0, labels, scores)
        return target

    def __getitem__(self, index):
        entry = self.entries[index]
        features, spatials = self.load_image(entry['img_idx'])
        question = entry['question']
        question_id = entry['question_id']
        answer = entry['answer']
        return features, spatials, question, answer, question_id

    def __len__(self):
        return len(self.entries)
