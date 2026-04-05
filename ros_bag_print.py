import bagpy
from bagpy import bagreader
import os
from pathlib import Path
from tqdm import tqdm
# b = bagreader('gprbag/gpr_1228_2.bag')
name = 'gpr_1228_2.bag'

b = bagreader(f'gprbag/{name}')
print(b.topic_table)
imu_data = b.message_by_topic('/mavros/imu/data')
print(imu_data)
# imu_data.to_csv(f'{name}_imu_data.csv', index=False)
bag_dir = Path('gprbag')
for bag_file in tqdm(bag_dir.glob('*.bag')):
    b = bagreader(str(bag_file))
    imu_data = b.message_by_topic('/mavros/imu/data')
    # if imu_data is not None and not imu_data.empty:
    #     csv_name = f'{bag_file.stem}_imu_data.csv'
    #     imu_data.to_csv(csv_name, index=False)
    #     print(f'Saved {csv_name}')