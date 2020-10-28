import numpy as np

from cal_rot_matrix import cal_rotate_matrix, rotmat2expmap

def xyz2expmap(data):
    assert(data.shape[0] == 32 and data.shape[1] == 3)
    
    data = data[:,[0,2,1]]
    
    expmap = np.zeros((33, 3))
    expmap[0,:] = np.array([0,0,0])
    
    rot_1 = np.zeros((33, 3, 3))
    rot_2 = np.zeros((33, 3, 3))
    
    bone_direct = np.array([[0,0,0],[-1,0,0],[0,-1,0],[0,-1,0],[0,0,1],[0,0,1],
                           [1,0,0],[0,-1,0],[0,-1,0],[0,0,1],[0,0,1],
                           [0,0,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                           [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,0],[0,0,1],[0,1,0],[0,0,0],
                           [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,0],[0,0,1],[0,1,0],[0,0,0]])
    
    d_struct = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31])
    
    parent = d_struct - 1

    parent_rotation = []
    
    for i in range(32):
        
        if(parent[i] == -1 or (bone_direct[i,0] == 0 and bone_direct[i,1] == 0 and bone_direct[i,2] == 0)): 
            expmap[i + 1] = rotmat2expmap(np.array([[1,0,0],[0,1,0],[0,0,1]]))
            rot_1[i] = np.array([[1,0,0],[0,1,0],[0,0,1]])
            rot_2[i] = np.array([[1,0,0],[0,1,0],[0,0,1]])
        else: 
            xyz = data[i]
            
            p_xyz = data[parent[i]]
            
            bone_xyz = xyz - p_xyz
            bone_len = np.sqrt(np.sum(bone_xyz ** 2))
            
            origin_xyz = bone_direct[i] * bone_len
            
            rot_matrix = cal_rotate_matrix(origin_xyz, bone_xyz)
            
            rot_2[i] = rot_matrix
            
            #print('rot_2[{}]\n{}'.format(parent[i], rot_2[parent[i]]))
            rot_1[parent[i]] = rot_matrix.dot(np.matrix(rot_2[parent[i]]).I)
            
            expmap[d_struct[i]] = rotmat2expmap(rot_1[parent[i]])
        
    return expmap
