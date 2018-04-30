import random
import json
DEV_SIZE = 100

dev_samples = random.sample(range(1, 800), DEV_SIZE)
print("Indices", len(dev_samples))

with open('train_student.json') as f:
    data = json.load(f)

print(len(data['data']))

dev_data = {'data': []}
train_data = {'data': []}

dev = [data['data'][index] for index in dev_samples]
print("Dev size", len(dev))
train = [sample for index, sample in enumerate(data['data']) if index not in dev_samples]
print("Train size", len(train))

dev_data['data'] = dev
train_data['data'] = train

with open('student_train.json', 'w') as f:
    json.dump(train_data, f)

with open('student_dev.json', 'w') as f:
    json.dump(dev_data, f)
