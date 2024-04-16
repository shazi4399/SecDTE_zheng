from common.helper import *
from common.constants import *
from parties.partySelect import SelectParty
from parties.partyEval import EvalParty
from fss import ICNew
import sycret
import numpy as np
from benchmarkOption import *
import pickle
#TODO: 
# 1. generate multiple instances of aux values
class Player:
    def __init__(self, id ,networkPool):
        self.ID = id
        self.network = networkPool
        self.resetMsgPoolAsList()
        self.eqFSS = sycret.EqFactory(n_threads=6)
        self.IntCmp = ICNew()

        self.prg1 = PRF(
                (SEED_KEYS[id][0]+"randomSeed").encode('ascii'), VEC_VAL_MAX_BOUND)
        # shared PRG with right neighbour
        self.prg2 = PRF(
            (SEED_KEYS[id][1]+"randomSeed").encode('ascii'), VEC_VAL_MAX_BOUND)
    
    def getZeroShares(self,msg,bound):
        a1 = self.prg1(msg.encode('ascii'), 1)[0]%bound
        a2 = self.prg2(msg.encode('ascii'), 1)[0]%bound
        return (a1-a2+bound)%bound

    def getNodesAmount(self):
        return self.nodesAmount
    
    def resetMsgPoolAsList(self):
        self.msgPool={
            str( (self.ID+1)%3 ):[],
            str( (self.ID+2)%3 ):[]
        }
    
    def resetMsgPoolWithKeys(self,key0,key1):
        self.msgPool={
            str( (self.ID+1)%3 ):
                {
                    key0:[],
                    key1:[]
                }
            ,
            str( (self.ID+2)%3 ):
                {
                    key0:[],
                    key1:[]
                }
        }

    def resetMsgPoolWithCmpKeys(self,key0,key1):
        self.msgPool[str( (self.ID+1)%3 )] ={
                    key0:[ i for i in range(self.nodesAmount) ],
                    key1:[ i for i in range(self.nodesAmount) ]
                }

        self.msgPool[str( (self.ID+2)%3 )] ={
                    key0:[ i for i in range(self.nodesAmount) ],
                    key1:[ i for i in range(self.nodesAmount) ]
                }
            

    async def distributeNetworkPool(self):
        for key,val in self.msgPool.items():
            print("key:",key)
            print("val:",val)
            _type = type(val).__name__
            willSend=False
            if _type == "dict":
                for inKey,inVal in val.items():
                    if len(inVal)>0:
                        willSend=True
            elif _type == "list":
                if len(val)>0:
                    willSend=True
            if willSend:
                self.network.asend("P"+key,val)
    
    async def distributeNetworkPoolConservely(self):
        for key,val in self.msgPool.items():
            _type = type(val).__name__
            willSend=False
            if _type == "dict":
                for inKey,inVal in val.items():
                    if len(inVal)>0:
                        willSend=True
            elif _type == "list":
                if len(val)>0:
                    willSend=True
            if willSend:
                await self.network.send("P"+key,val)
                
    
    async def distributeScatteredNetworkPool(self):
        ''' 
        Underlying procedure:
         0. send samll chunks instead of sending a long message at once
         1. use corontines when distributing messages to different servers
        '''
        chunks_size = 0
        willSendKeys = []
        # Prepare the message
        print("msgPool:",self.msgPool)
        for key, val in self.msgPool.items():
            _type = type(val).__name__
            if _type == "dict":
                for _, inVal in val.items():
                    total_size = len(inVal)
                    if total_size>0:
                        chunks_size = int( total_size/PERFORMANCE_BATCH_SIZE)
                        if total_size%PERFORMANCE_BATCH_SIZE >0:
                            chunks_size += 1
                        willSendKeys.append(key)
                        break
                    
            elif _type == "list":
                total_size = len(val)
                if total_size > 0:
                    willSendKeys.append(key)
                    chunks_size = int(total_size/PERFORMANCE_BATCH_SIZE)
                    if total_size % PERFORMANCE_BATCH_SIZE > 0:
                        chunks_size += 1
        
        #However, we need to sync each small chunk
        print("chunk size is: ",chunks_size)
        for i in range(chunks_size):
            for key in willSendKeys:
                cur_data = self.msgPool.get(key)
                _type = type(cur_data).__name__
                if _type == "dict":
                    new_map = {}
                    for mapKey, value in cur_data.items():
                        if i < chunks_size-1:
                            new_map[mapKey] = value[i *PERFORMANCE_BATCH_SIZE:(i+1)*PERFORMANCE_BATCH_SIZE]
                        else:
                            new_map[mapKey] = value[i*PERFORMANCE_BATCH_SIZE:]
                    self.network.asend("P"+key, [i, new_map])
                    # last_corontines.append( self.network.send("P"+key, new_map) )
                elif _type == "list":
                    if i < chunks_size-1:
                        self.network.asend(
                            "P"+key, [i, cur_data[i * PERFORMANCE_BATCH_SIZE:(i+1)*PERFORMANCE_BATCH_SIZE]])
                    else:
                        self.network.asend( "P"+key, [i, cur_data[i*PERFORMANCE_BATCH_SIZE:]])
        

    def inputASS(self, leafSS, vSS, treeSS, condShare, selectMatShare, sharesVec2Mat):
        self.leafSS = leafSS
        self.leafNodesAmount = len(self.leafSS)
        self.selectMatShare = selectMatShare
        self.sharesVec2Mat = sharesVec2Mat

        self.vSS = vSS
        self.tVecDim = len(self.vSS)
        self.idxSS = treeSS[0]
        self.thresholdSS = treeSS[1]
        self.condShares = condShare
        self.nodesAmount = len(condShare)


    def inputFSSKeys(self,fssKeys):
        self.fssKeys = fssKeys

    def inputBeaverTriples(self, beavers):
        self.beavers = beavers

    def inputbool_BeaverTriples(self, bool_beavers):
        self.bool_beavers = bool_beavers

    def input0Shares(self,zeroShares):
        self.zeroShare = zeroShares

    def input0matShares(self,zeroMatShares):
        self.zeroMatShares =zeroMatShares
    def inputpc(self,pc):
        self.pc = pc
    def inputPath(self, pathpub):
        self.pathPub = pathpub
    def inputXY(self, x, y):
        self.x_share = x
        self.y_share = y
    def inputbool_XY(self, bool_x, bool_y):
        self.bool_x_share = bool_x
        self.bool_y_share = bool_y
    def inputmat_XY(self, mat_x_share, mat_y_share):
        self.mat_x_share = mat_x_share
        self.mat_y_share = mat_y_share
    def inputmat_beavers(self, mat_beavers):
        self.mat_beavers = mat_beavers

    def inputxvector(self, xvector):
        self.xvector = xvector
    def inputbit_decomposition(self, x):
        binary_zero = [0] * BINARY_REPRESENTED
        binary_x = decimal_to_binary_32(x)
        print("binary_x:",binary_x)

        self.y = binary_x
        if self.ID == 0:
            self.a = binary_x
            self.b = binary_zero
        else:
            self.b = binary_x
            self.a = binary_zero
        self.d = [0] * BINARY_REPRESENTED
        self.e = [0] * BINARY_REPRESENTED
        self.c = [0] * BINARY_REPRESENTED
        self.x = [0] * BINARY_REPRESENTED
    def inputcompare_res(self, b):
        self.compare_res = b

    def mul(self):
        # print("self.beavers: ",self.beavers)
        # print("self.x_share: ",self.x_share)
        # print("self.y_share: ",self.y_share)
        self.e_share = self.x_share - self.beavers[0]
        self.f_share = self.y_share - self.beavers[1]
        ef_share_bytes = pickle.dumps([self.e_share, self.f_share])
        if self.ID == 0:
            self.network.asend("P1", ef_share_bytes)
            # message = self.network.recv("P1")
        else:
            #message = self.network.recv("P0")
            self.network.asend("P0", ef_share_bytes)

    def bool_mul(self):
        self.bool_e_share = self.bool_x_share ^ self.bool_beavers[0]
        self.bool_f_share = self.bool_y_share ^ self.bool_beavers[1]
        ef_share_bytes = pickle.dumps([self.bool_e_share, self.bool_f_share])
        if self.ID == 0:
            self.network.asend("P1", ef_share_bytes)
        else:
            self.network.asend("P0", ef_share_bytes)

    def mat_mul(self):
        print("self.mtaX:", self.mat_x_share)
        print("self.matY:", self.mat_y_share)
        self.mat_eshare = matrix_subtraction(self.mat_x_share, self.mat_beavers[0])
        print("self.mat_eshare:", self.mat_eshare)
        self.mat_fshare = matrix_subtraction(self.mat_y_share, self.mat_beavers[1])
        print("self.mat_fshare:", self.mat_fshare)
        ef_share_bytes = pickle.dumps([self.mat_eshare, self.mat_fshare])
        # # 重新加载字节对象为原始对象
        # print("ef_share_bytes:", ef_share_bytes)
        # loaded_objects = pickle.loads(ef_share_bytes)
        #
        # # 现在 loaded_objects 是一个包含了之前打包的两个对象的列表
        # # 可以根据需要从 loaded_objects 中获取这些对象
        # mat_eshare = loaded_objects[0]
        # mat_fshare = loaded_objects[1]
        # print("mat_eshare111:", mat_eshare)
        # print("mat_fshare111:", mat_fshare)
        if self.ID == 0:
            self.network.asend("P1", ef_share_bytes)
        else:
            self.network.asend("P0", ef_share_bytes)

    def getdiff(self):
        # 使用列表推导式逐元素相减
        self.diff = [x - y for x, y in zip(self.xvector, self.thresholdSS)]


    def genres_pathcost(self):
        pathcosts = []
        for i in range(len(self.pathPub)):
            pathcost = []
            #print("self.pathPub:",self.pathPub)
            for j in range(len(self.pathPub[i])):
                if(j == len(self.pathPub[i])-1):
                    continue
                if(self.pathPub[i][j+1] == 2*(self.pathPub[i][j])+1): # left  left = [b]
                    #print("self.pathPub[i][j]:",self.pathPub[0][j])
                    pathcost.append(self.compare_res[j])
                else:# right_out = 1-[b]
                    if(self.ID == 0):
                        pathcost.append(1-self.compare_res[j])
                    else:
                        pathcost.append(self.compare_res[j])
            #print("pathcost:",pathcost)
            pathcosts.append(pathcost)
        #print("pathcosts", pathcosts)
        self.pathcosts = pathcosts

    def gen_eval_res(self):
        pc = self.pc.copy()
        #v = self.leafSS.copy()
        # 使用列表推导式和 zip() 函数相加
        v = [x + (y*random.randint(0,VEC_VAL_MAX_BOUND)) for x, y in zip(self.leafSS, pc)]
        pc_star = permuat_vector(pc,1)
        v_star = permuat_vector(v,1)
        pc_v_star = pickle.dumps([pc_star, v_star])
        if self.ID == 0:
            self.network.asend("P1", pc_v_star)
        else:
            self.network.asend("P0", pc_v_star)
        self.pc_star = pc_star
        self.v_star = v_star
    # Collect messages to be sent for FSS evaluation
    def compare0(self):
        # Initialize some comparison related lists
        self.valsPQ_shares=[i for i in range(self.nodesAmount)]
        self.cmpResultShares=[i for i in range(self.nodesAmount)]

        if self.ID == 0:
            for i,select in enumerate(self.selectedShares):
                # local convert (2,3)-RSS to (2,2)-SS   (0，1)，2）
                newSelectShare =  inRing(select[0]+select[1], INT_32_MAX)
                newThresholdShare = inRing( self.thresholdSS[i][0] + self.thresholdSS[i][1], INT_32_MAX)
                subShare = inRing( newSelectShare - newThresholdShare, INT_32_MAX)
                eqalRnd = inRing( subShare + self.fssKeys[i][1], INT_32_MAX)
                lessRnd = inRing(  subShare + self.fssKeys[i][3], INT_32_MAX)
                self.vals4Equal.append( eqalRnd )
                self.vals4Less.append( lessRnd )
                self.msgPool["1"].append( [eqalRnd,lessRnd] )

            
        elif self.ID == 1:
            for i,select in enumerate(self.selectedShares):
                # local convert (2,3)-RSS to (2,2)-SS
                newSelectShare = select[1]
                newThresholdShare = self.thresholdSS[i][1]
                subShare = inRing( newSelectShare - newThresholdShare, INT_32_MAX)

                eqalRnd = inRing( subShare + self.fssKeys[i][1], INT_32_MAX)
                lessRnd = inRing( subShare + self.fssKeys[i][3], INT_32_MAX)
                self.vals4Equal.append( eqalRnd )
                self.vals4Less.append( lessRnd )
                self.msgPool["0"].append( [eqalRnd,lessRnd] )

    async def compare1(self,chunk_index,otherShares):
        if self.ID <= 1:
            piece_size = len(otherShares)
            for i in range(piece_size):
                whole_index = chunk_index*PERFORMANCE_BATCH_SIZE + i
                reveal =  inRing(self.vals4Equal[whole_index] + otherShares[i][0], INT_32_MAX)
                r_eq = self.eqFSS.eval(self.ID, np.array( [np.int64(reveal)] ), self.fssKeys[whole_index][0])
                r_eq = r_eq[0].item()#Convert numpy array to a normal int value

                reveal =  inRing(self.vals4Less[whole_index]+otherShares[i][1], INT_32_MAX)
                r_le = self.IntCmp.eval(self.ID, np.array( [np.int64(reveal)] ), self.fssKeys[whole_index][2])
                xor = inRing( r_eq + r_le, BOOLEAN_BOUND)

                cmpShare = [inRing( r_eq + self.getZeroShares("bRand",BOOLEAN_BOUND), BOOLEAN_BOUND),0]
                
                if self.ID == 0:#ShareConv RSS->SS
                    condShare = inRing( self.condShares[whole_index][0]+self.condShares[whole_index][1], BOOLEAN_BOUND)
                    alpha = inRing( self.triples[whole_index][0][0]+self.triples[whole_index][0][1], BOOLEAN_BOUND)
                    beta = inRing( self.triples[whole_index][1][0]+self.triples[whole_index][1][1], BOOLEAN_BOUND)
                    m_pVal = inRing( condShare + alpha, BOOLEAN_BOUND)
                    m_qVal = inRing( xor + beta, BOOLEAN_BOUND)

                    self.valsPQ_shares[whole_index] =  (m_pVal,m_qVal) 
                    self.msgPool["1"]["sc-and"][whole_index]= (m_pVal,m_qVal) 
                    self.msgPool["2"]["sc-and"][whole_index]= (m_pVal,m_qVal) 
                    self.msgPool["2"]["invConv"][whole_index]=cmpShare[0] 
                else:
                    condShare = self.condShares[whole_index][1]
                    alpha = self.triples[whole_index][0][1]
                    beta = self.triples[whole_index][1][1]
                    m_pVal = inRing( condShare + alpha, BOOLEAN_BOUND)
                    m_qVal = inRing( xor + beta, BOOLEAN_BOUND)

                    self.valsPQ_shares[whole_index] =  (m_pVal,m_qVal)
                    self.msgPool["0"]["sc-and"][whole_index]= (m_pVal,m_qVal)
                    self.msgPool["2"]["sc-and"][whole_index]= (m_pVal,m_qVal)
                    self.msgPool["0"]["invConv"][whole_index]=cmpShare[0] 
                self.cmpResultShares[whole_index]=cmpShare
        else:
            for i in range(self.nodesAmount):
                cmpShare = [self.getZeroShares("bRand",BOOLEAN_BOUND),0]
                self.msgPool["1"]["invConv"][i]= cmpShare[0]
                self.cmpResultShares[i]=cmpShare
    
    def compare2(self,pq_vals0,pq_vals1,otherBShareList):
        vec=[[],[]]
        for i in range(self.nodesAmount):
            if self.ID <= 1:
                P = inRing( self.valsPQ_shares[i][0] + pq_vals0[i][0], BOOLEAN_BOUND) 
                Q = inRing( self.valsPQ_shares[i][1] + pq_vals0[i][1], BOOLEAN_BOUND)
            else:
                P = inRing( pq_vals0[i][0] + pq_vals1[i][0], BOOLEAN_BOUND)
                Q = inRing( pq_vals0[i][1] + pq_vals1[i][1], BOOLEAN_BOUND)
            cmpShare = self.triples[i][2]
            cmpShare = RSS_local_add(cmpShare, RSS_local_mul( self.triples[i][0],Q, BOOLEAN_BOUND),BOOLEAN_BOUND)
            cmpShare = RSS_local_add(cmpShare, RSS_local_mul( self.triples[i][1],P, BOOLEAN_BOUND),BOOLEAN_BOUND)
            if self.ID == 0:
                cmpShare = RSS_local_add(cmpShare,[0,P*Q], BOOLEAN_BOUND)
            elif self.ID == 1:
                cmpShare = RSS_local_add(cmpShare,[P*Q,0], BOOLEAN_BOUND)
            
            self.cmpResultShares[i][1] =  otherBShareList[i]
            tmp = RSS_local_add(cmpShare,self.cmpResultShares[i], BOOLEAN_BOUND)
            for j in range(2):
                vec[j].append(tmp[j])
        # Reshape self.cmpResultShares from m x 2 to 2 x m
        self.cmpResultShares = vec
        print("self.comResultShares :",self.cmpResultShares)
    # Embed shuffle reveal within this function
    def pathEval0_shuffleReveal(self,bShares,piShares):
        xorShares = []
        for j in range(2):
            xorShares.append( vec_add_withBound(self.cmpResultShares[j], bShares[j], BOOLEAN_BOUND) )
        
        seed = 12
        dim = len(bShares[0])
        party = EvalParty(self.ID, dim, seed)
        alpha = party.genReplicatedVecShare("alphaRnd")
        #shuffleReveal
        if self.ID ==0:
            beta1 = vec_add_withBound(xorShares[0], xorShares[1], BOOLEAN_BOUND)
            pi_1 = piShares[0]
            sigma = vec_sub(shuffleNonLeaf(beta1, pi_1), alpha[0])
            self.msgPool["1"]["shuffleReveal"] =  sigma 
        elif self.ID == 2:
            pi_1 = piShares[1]
            beta3 = xorShares[0]
            gamma = vec_add(shuffleNonLeaf(beta3, pi_1), alpha[1])
            self.msgPool["1"]["shuffleReveal"]=  gamma 

    def pathEval0_optShuffle(self,piShares):
        #seed = random.randint(0, 25)
        seed = 13
        party = EvalParty(self.ID, self.leafNodesAmount, seed)
        alpha = party.genReplicatedVecShare("alphaRnd")
        if self.ID == 0:#P0 to P2
            beta1 = vec_add(self.leafSS[0], self.leafSS[1])
            pi_1 = piShares[0]
            pi_2 = piShares[1]
            sigma = vec_sub(permute(pi_1,beta1),alpha[0])
            sigma = vec_sub(permute(pi_2, sigma), alpha[1])
            self.msgPool["2"]["optShuffle"]= sigma
        elif self.ID == 2:#P2 to P1
            beta3 = self.leafSS[0]
            pi_1 = piShares[1]
            gamma = vec_add(permute(pi_1, beta3),alpha[1])
            self.msgPool["1"]["optShuffle"]= gamma

    # shuffleReveal responded by p1
    def pathEval1_shuffleRevealRespond(self,sigma,gamma,piShares):
        pi_2 = piShares[0]
        pi_3 = piShares[1]
        out = vec_add_withBound(gamma, sigma, BOOLEAN_BOUND)
        out = shuffleNonLeaf(out, pi_2)
        out = shuffleNonLeaf(out, pi_3)
        self.shuffleReveal = list(out)
        self.msgPool["0"]["shuffleReveal"] = list(out)
        self.msgPool["2"]["shuffleReveal"] = list(out)
        # print("out value is:", out)
        # for v in out:
        #     self.msgPool["0"]["shuffleReveal"].append(v)
        #     self.msgPool["2"]["shuffleReveal"].append(v)
        
    def pathEval1_optShuffleRespond(self,gamma,sigma,piShares):
        seed = 13
        party = EvalParty(self.ID, self.leafNodesAmount, seed)
        alpha = party.genReplicatedVecShare("alphaRnd")

        reshapeRnd = party.genZeroShare("randomRnd",self.leafNodesAmount, VEC_VAL_MAX_BOUND)
        if self.ID == 0:
            self.shuffledShares.append( reshapeRnd )
            self.msgPool["2"]["optShuffle"] = reshapeRnd
        elif self.ID == 1:#(P2 to P1:gamma), P1 now Responds
            pi_2 = piShares[0]
            pi_3 = piShares[1]
            out1 = vec_add(permute(pi_2,gamma),alpha[0])
            out1 = permute(pi_3,out1)
            self.shuffledShares.append( vec_add(reshapeRnd,out1) )
            self.msgPool["0"]["optShuffle"] = self.shuffledShares[0]
        elif self.ID == 2:#(P0 to P2:sigma), P2 now Responds
            pi_3 = piShares[0]
            out2 = permute(pi_3, sigma)
            self.shuffledShares.append( vec_add(reshapeRnd,out2) )
            self.msgPool["1"]["optShuffle"] = self.shuffledShares[0]
    
    def pathEval2(self,revealVec,otherShuffleShare):
        self.shuffledShares.append( otherShuffleShare )
        if self.ID != 1:
            self.shuffleReveal = revealVec
        index = getEvalIndex(self.shuffleReveal)
        # print("final index is: ",index)
        v0 = self.shuffledShares[0][index]
        v1 = self.shuffledShares[1][index]
        return v0,v1

    def pathEvalPoly0(self):
        pathpolys_i = []
        pathpolys_i1 = []
        print()
        for i in range(len(self.pathPub)):
            pathpoly_i = []
            pathpoly_i1 = []
            #print("self.pathPub:",self.pathPub)
            for j in range(len(self.pathPub[i])):
                if(j == len(self.pathPub[i])-1):
                    continue
                if(self.pathPub[i][j+1] == 2*(self.pathPub[i][j])+1): # left  left = [b]
                    #print("self.pathPub[i][j]:",self.pathPub[0][j])
                    pathpoly_i.append(self.cmpResultShares[0][j])
                    pathpoly_i1.append(self.cmpResultShares[1][j])
                else:# right_out = 1-[b]
                    if(self.ID == 0):
                        pathpoly_i.append(1-self.cmpResultShares[0][j])
                        pathpoly_i1.append((-1)*self.cmpResultShares[1][j])
                    elif(self.ID == 1):
                        pathpoly_i.append((-1)*self.cmpResultShares[0][j])
                        pathpoly_i1.append((-1)*self.cmpResultShares[1][j])
                    else:
                        pathpoly_i.append((-1)*self.cmpResultShares[0][j])
                        pathpoly_i1.append(1-self.cmpResultShares[1][j])
            #print("pathpoly_i:",pathpoly_i)
            #print("pathpoly_i1:",pathpoly_i1)
            pathpolys_i.append(pathpoly_i)
            pathpolys_i1.append(pathpoly_i1)
        #print("pathpolys_i", pathpolys_i)
        #print("pathpolys_i1", pathpolys_i1)
        self.pathpolys = [pathpolys_i, pathpolys_i1]

    def pathEvalPoly1(self,pathID):
        pass

