import numpy as np
import torch


def findSalient(x, All_indices, adj, sumsMap, device=None):
    dist = torch.zeros(adj.shape[1], device=device)
    adj = torch.from_numpy(adj).to(device)
    #for k in range(adj.shape[1]):
        #dist[k] = 2 - 2 * np.matmul(np.transpose(x)[adj[0,k],:], x[:, adj[1,k]]) + 1e-9
    dist = 2 - 2 * torch.sum(x[:,adj[0,:]] * x[:, adj[1,:]],axis=0) + 1e-9
    dist = torch.abs(dist)  #to stop ALL NaNs! DO NOT REMOVE this line!
    dist = torch.sqrt(dist)
    #if there are ever any NaNs, training will fail!

# now gotta work out which elements of dist need to be summed together
# i.e. 0 = 1+6+7, 1 = 0+2+6+7+8

    if torch.isnan(dist).any():
        print(torch.isnan(dist).any())

    saliency = torch.zeros(len(sumsMap), device=device)
    for iter, item in enumerate(sumsMap):
        #subtotal = 0
        #k = 0

        saliency[iter] = torch.sum(dist[torch.tensor(item)])/len(item)
        #for subitem in item:
        #    subtotal += dist[subitem]
        #    k+=1
        #saliency[iter] = subtotal/k

# then divide dist by number of sums

    #not sure if this is a good idea or not:
    saliency = saliency/saliency.max()

# by this point should have a vector of length numRegions
# this vector is then used, with a threshold, to determine the bland regions.
    return saliency

def torch_nn_attention(x, y, xKeepBool, yKeepBool):

    y = torch.transpose(y,0,1)
    mul = torch.matmul(y[yKeepBool,:], x[:,xKeepBool])  # (B,M,N)

    dist = 2 - 2 * mul
    dist[dist < 1e-9] = 1e-9
    dist = torch.sqrt(dist)

    fw_dist, ind = torch.min(dist, 1)
    ind_b = torch.argmin(dist, 0)
#note: ind is reference -> query and ind_b is query -> reference
#i.e. ind is the matching position in query for each reference region
# ind_b is the matching position in reference for each query region
    return dist, fw_dist, ind, ind_b


def attn_calc_indices(regionSize, moveAmount):
    H = 480 / 16  #this will always be 16 for vgg-16
    W = 640 / 16
    paddingSize = [0, 0]
    region_size = (regionSize, regionSize)
    move_amount = (moveAmount, moveAmount)

    Hout = (H + (2 * paddingSize[0]) - region_size[0]) / move_amount[0] + 1
    Wout = (W + (2 * paddingSize[1]) - region_size[1]) / move_amount[1] + 1

    Hout = int(Hout)
    Wout = int(Wout)

    numRegions = Hout * Wout

    k = 0
    kk = 0
    founds = []
    sumsMap = []

    # All_indices = np.zeros((2, numRegions), dtype=int)
    adj = np.zeros((2, 10000), dtype=int)
    for i in range(0, Hout):
        for j in range(0, Wout):
            # All_indices[0, k] = j  # x,y coordinates
            # All_indices[1, k] = i

            sumsMap.append([])

            #I discovered (SH, on 220920) that NetVLAD is encoding spatial data to a much greater extent than
            #previously thought. A patch of sky will have a different descriptor to another visually identical
            #patch of sky located elsewhere in the image!

            #we notice that the positional height (which is a proxy for depth in monocular) is the greatest
            #spatial learnt factor.
            #therefore, we adjust the saliency algorithm to only consider left-and-right side regions.

            # now calc. adjacency list:
            foundBool = False
            if (i - 1 >= 0) and (j - 1 >= 0):  # top-left
                for iter, item in enumerate(founds):
                    if k == item[1] and (k - Wout - 1) == item[0]:  # the flip case will be in the opp. order
                        foundBool = True
                        sumsMap[k].append(iter)
                if not foundBool:
                    adj[0, kk] = k
                    adj[1, kk] = k - Wout - 1
                    founds.append([])
                    founds[kk].append(k)
                    founds[kk].append(k - Wout - 1)
                    sumsMap[k].append(kk)
                    kk += 1
            # adj[k].append(k-Wout-1)
            foundBool = False
            if (i - 1 >= 0):  # top
                for iter, item in enumerate(founds):
                    if k == item[1] and (k - Wout) == item[0]:  # the flip case will be in the opp. order
                        foundBool = True
                        sumsMap[k].append(iter)
                if not foundBool:
                    adj[0, kk] = k
                    adj[1, kk] = k - Wout
                    founds.append([])
                    founds[kk].append(k)
                    founds[kk].append(k - Wout)
                    sumsMap[k].append(kk)
                    kk += 1
            # adj[k].append(k-Wout)
            foundBool = False
            if (i - 1 >= 0) and (j + 1 < Wout):  # top-right
                for iter, item in enumerate(founds):
                    if k == item[1] and (k - Wout + 1) == item[0]:  # the flip case will be in the opp. order
                        foundBool = True
                        sumsMap[k].append(iter)
                if not foundBool:
                    adj[0, kk] = k
                    adj[1, kk] = k - Wout + 1
                    founds.append([])
                    founds[kk].append(k)
                    founds[kk].append(k - Wout + 1)
                    sumsMap[k].append(kk)
                    kk += 1
            # adj[k].append(k-Wout+1)
            foundBool = False
            if (j - 1 >= 0):  #left
                for iter, item in enumerate(founds):
                    if k == item[1] and (k - 1) == item[0]:  # the flip case will be in the opp. order
                        foundBool = True
                        sumsMap[k].append(iter)
                if not foundBool:
                    adj[0, kk] = k
                    adj[1, kk] = k - 1
                    founds.append([])
                    founds[kk].append(k)
                    founds[kk].append(k - 1)
                    sumsMap[k].append(kk)
                    kk += 1
            # adj[k].append(k-1)
            foundBool = False
            if (j + 1 < Wout):  #right
                for iter, item in enumerate(founds):
                    if k == item[1] and (k + 1) == item[0]:  # the flip case will be in the opp. order
                        foundBool = True
                        sumsMap[k].append(iter)
                if not foundBool:
                    adj[0, kk] = k
                    adj[1, kk] = k + 1
                    founds.append([])
                    founds[kk].append(k)
                    founds[kk].append(k + 1)
                    sumsMap[k].append(kk)
                    kk += 1
            # adj[k].append(k+1)
            foundBool = False
            if (i + 1 < Hout) and (j - 1 >= 0):  # bottom-left
                for iter, item in enumerate(founds):
                    if k == item[1] and (k + Wout - 1) == item[0]:  # the flip case will be in the opp. order
                        foundBool = True
                        sumsMap[k].append(iter)
                if not foundBool:
                    adj[0, kk] = k
                    adj[1, kk] = k + Wout - 1
                    founds.append([])
                    founds[kk].append(k)
                    founds[kk].append(k + Wout - 1)
                    sumsMap[k].append(kk)
                    kk += 1
            # adj[k].append(k+Wout-1)
            foundBool = False
            if (i + 1 < Hout):  # bottom
                for iter, item in enumerate(founds):
                    if k == item[1] and (k + Wout) == item[0]:  # the flip case will be in the opp. order
                        foundBool = True
                        sumsMap[k].append(iter)
                if not foundBool:
                    adj[0, kk] = k
                    adj[1, kk] = k + Wout
                    founds.append([])
                    founds[kk].append(k)
                    founds[kk].append(k + Wout)
                    sumsMap[k].append(kk)
                    kk += 1
            # adj[k].append(k+Wout)
            foundBool = False
            if (i + 1 < Hout) and (j + 1 < Wout):  #bottom-right
                for iter, item in enumerate(founds):
                    if k == item[1] and (k + Wout + 1) == item[0]:  # the flip case will be in the opp. order
                        foundBool = True
                        sumsMap[k].append(iter)
                if not foundBool:
                    adj[0, kk] = k
                    adj[1, kk] = k + Wout + 1
                    founds.append([])
                    founds[kk].append(k)
                    founds[kk].append(k + Wout + 1)
                    sumsMap[k].append(kk)
                    kk += 1
            # adj[k].append(k+Wout+1)

            k += 1
    # now remove the ending zeros from the pre-alloc:
    adj = adj[:, :kk]
    # later adj will be looped through, and the resulting difference scores calculated

    return adj, sumsMap, Hout, Wout


#example calling code:
    # for j in range(dbfeats.shape[0]):
    #     saliency = findSalient(torch.from_numpy(dbfeats[j,:,:]).to(device), All_indices, adj, sumsMap)
    #     saliency = saliency.cpu().numpy()
    #     dbKeepBools.append([])
    #     dbKeepBools[j].append(np.argwhere(saliency>0.4))

# if opt.useAttention:  # calc the attention score of the query
#     saliency = findSalient(qfgpuo, All_indices, adj, sumsMap, device=device)
#     saliency = saliency.cpu().numpy()
#     # to plot the saliency:
#     # plotS = np.reshape(saliency,(26,36))
#
#     qKeepBool = np.argwhere(saliency > 0.4)
#     qKeepBool = np.reshape(qKeepBool, -1)


# diff1[qIx, k], diff2[qIx, k], hMat, _, _ = compare_two_ransac(
#     qfgpu, dbfeat, device, All_indices, Hout, Wout, qKeepBool, dbKeepBool)
# hMatList.append(hMat)



