# type 'cp' or '' (VQA-CP or VQA)
type = ''
# dataset version (v1 or v2)
version = 'v2'
# 'train' or 'trainval'
mode = 'train'

# import settings
task = 'OpenEnded'
dataset = 'mscoco'
max_question_len = 14
visual_glimpses = 1
test_split = 'test2015'  # either 'test-dev2015' or 'test2015'

# directory containing all images
dataroot = '/home/share/liuyibing/vqa/'
if type == 'cp':
    image_path = dataroot + 'mscoco-cp'
else:
    image_path = dataroot + 'mscoco'

# directory containing the question and annotation jsons
if version == 'v1':
    qa_path = dataroot + 'vqa-{}1.0/qa_path/'.format(type)
elif version == 'v2':
    qa_path = dataroot + 'vqa-{}2.0/qa_path/'.format(type)

glove_path = dataroot + 'word_embed/glove/'

# dataroot and proceed_data path
main_path = '../data/'
cache_root = main_path + (type + version)
rcnn_path = main_path + '../rcnn-data/'

output_features = 2048
rcnn_output_size = 36  # max number of object proposals per image
bottom_up_trainval_path = dataroot + 'trainval_{}'.format(rcnn_output_size)  # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = dataroot + 'test2015_{}'.format(rcnn_output_size)  # directory containing the .tsv file(s) with bottom up features
rcnn_trainval_path = rcnn_path + 'trainval_{}.h5'.format(rcnn_output_size)  # path where preprocessed features from the trainval split are saved to and loaded from
rcnn_test_path = rcnn_path + 'test_{}.h5'.format(rcnn_output_size)  # path where preprocessed features from the test split are saved to and loaded from

hid_dim = 1024
epochs = 30
workers = 4

margin = 0
lamda = 0.1

w = 50
c = 0.1
masks = 3
lr = 2e-3
