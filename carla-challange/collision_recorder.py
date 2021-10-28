import os
import sys



def record(data):
    file1 = open(data_folder + "collision_data.txt", "a")
    file1.writelines(self.client.show_recorder_collisions(data_folder + "collision_data.txt", "v", "a"))
    # Closing file
    #file1.close()





if __name__ == '__main__':
        data = "/home/scope/Carla/carla-dockers/simulation_data/collision_data.log"
        record(data)
