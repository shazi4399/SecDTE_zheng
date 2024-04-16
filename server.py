from tno.mpc.communication import Pool
import time
import sys
import asyncio
from common.helper import *
from benchmarkOption import *
from dealer import Dealer
from player import Player
import pickle
async def async_main(_id):
    # Create the network pool for the current server
    pool = Pool()

    pool.add_http_server(addr=SERVER_IPS[_id], port=NEWTWORK_PORTS[_id])
    pool.add_http_client(
        "P" + str((_id + 1) % 2),
        addr=SERVER_IPS[(_id + 1) % 2],
        port=NEWTWORK_PORTS[(_id + 1) % 2],
    )

    # Setup the network connection
    if _id == 0:
        message = await pool.recv("P1")
        await pool.send("P1", "Hello!，I am P0")
        print(message)
    elif _id == 1:
        await pool.send("P0", "Hello!，I am P1")
        message = await pool.recv("P0")
        print(message)
    ### NO USE>>>>>>> ###
    if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
        start_time = time.time()
        initial_start_time = start_time
    if BENCHMARK_MEASURE_ONLINE_COMMU:
        lastRecv = pool.http_server.total_bytes_recv
    ### <<<<< 0. Setup phase ###
    dealer = Dealer()
    leafShares, sharesSampleVec, treeShares, condShare, selectMatShare, sharesVec2Mat, pathpub = dealer.getInputData(_id)
    print("leafShares:", leafShares)
    player = Player(_id, pool)
    player.inputASS(leafShares, sharesSampleVec, treeShares, condShare, selectMatShare,sharesVec2Mat)
    ### 0. Setup phase >>>>> ###

    ### <<< mul ###
    # z = x*y = 3*6 = 18
    x = [1,2]
    y = [2,4]
    beavers = dealer.distributeBeaverTriple()
    print("beavers:",beavers)
    player.inputBeaverTriples(beavers[_id])
    player.inputXY(x[_id],y[_id])
    player.mul()
    if _id == 0:
        message = await pool.recv("P1")
    else:
        message = await pool.recv("P0")
    ef = pickle.loads(message)
    e = player.e_share + ef[0]
    f = player.f_share + ef[1]
    z_share = (e * f * _id) + (f * player.beavers[0]) + \
            (e * player.beavers[1]) + player.beavers[2]
    print("z_share:",z_share)
    ### mul >>> ###

    ### <<< bool_mul ###
    # z = x*y = 1*1 = 1
    bool_x = [1,0]
    bool_y = [1,0]
    bool_beavers = dealer.distributebool_BeaverTriple()
    player.inputbool_BeaverTriples(bool_beavers[_id])
    player.inputbool_XY(bool_x[_id],bool_y[_id])
    player.bool_mul()
    if _id == 0:
        message = await pool.recv("P1")
    else:
        message = await pool.recv("P0")
    ef = pickle.loads(message)
    bool_e = player.bool_e_share^ef[0]
    bool_f = player.bool_f_share^ef[1]
    bool_z_share = (bool_e * bool_f * _id) ^ (bool_f * player.bool_beavers[0]) ^ \
            (bool_e * player.bool_beavers[1]) ^ player.bool_beavers[2]
    print("bool_z_share:",bool_z_share)
    ### bool_mul >>> ###

    if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
        start_time = time.time()
        #initial_start_time = start_time

    ### <<< bit_decomposition ###
    # BitDecompose
    # 实现将算数秘密共享 转换为二进制秘密共享
    x = [-3,1] #-2   （2^l）-》 2
    player.inputbit_decomposition(x[_id])
    player.inputbool_XY(player.a[0], player.b[0])
    player.bool_mul()
    if _id == 0:
        message = await pool.recv("P1")
    else:
        message = await pool.recv("P0")
    ef = pickle.loads(message)
    bool_e = player.bool_e_share^ef[0]
    bool_f = player.bool_f_share^ef[1]
    bool_z_share = (bool_e * bool_f * _id) ^ (bool_f * player.bool_beavers[0]) ^ \
            (bool_e * player.bool_beavers[1]) ^ player.bool_beavers[2]
    player.c[0] = bool_z_share
    player.x[0] = player.y[0]

    for i in range(1, BINARY_REPRESENTED):
        # a) [d] = [a]*[b]+1
        player.inputbool_XY(player.a[i], player.b[i])
        player.bool_mul()
        if _id == 0:
            message = await pool.recv("P1")
        else:
            message = await pool.recv("P0")
        ef = pickle.loads(message)
        bool_e = player.bool_e_share ^ ef[0]
        bool_f = player.bool_f_share ^ ef[1]
        bool_z_share = (bool_e * bool_f * _id) ^ (bool_f * player.bool_beavers[0]) ^ \
                       (bool_e * player.bool_beavers[1]) ^ player.bool_beavers[2]
        if _id == 0:
            player.d[i] = bool_z_share^0
        else:
            player.d[i] = bool_z_share^1

        # b) [e] = [y]*[c_(i-1)]+1
        player.inputbool_XY(player.y[i], player.c[i-1])
        player.bool_mul()
        if _id == 0:
            message = await pool.recv("P1")
        else:
            message = await pool.recv("P0")
        ef = pickle.loads(message)
        bool_e = player.bool_e_share ^ ef[0]
        bool_f = player.bool_f_share ^ ef[1]
        bool_z_share = (bool_e * bool_f * _id) ^ (bool_f * player.bool_beavers[0]) ^ \
                       (bool_e * player.bool_beavers[1]) ^ player.bool_beavers[2]
        if _id == 0:
            player.e[i] = bool_z_share^0
        else:
            player.e[i] = bool_z_share^1

        # c) [c] = [e]*[d]+1
        player.inputbool_XY(player.e[i], player.d[i])
        player.bool_mul()
        if _id == 0:
            message = await pool.recv("P1")
        else:
            message = await pool.recv("P0")
        ef = pickle.loads(message)
        bool_e = player.bool_e_share ^ ef[0]
        bool_f = player.bool_f_share ^ ef[1]
        bool_z_share = (bool_e * bool_f * _id) ^ (bool_f * player.bool_beavers[0]) ^ \
                       (bool_e * player.bool_beavers[1]) ^ player.bool_beavers[2]
        if _id == 0:
            player.c[i] = bool_z_share^0
        else:
            player.c[i] = bool_z_share^1

        # d) [x] = [y]+[c_(i-1)]
        player.x[i] = player.y[i] ^ player.c[i - 1]
    print("player.x:",player.x)

    # if BENCHMARK_MEASURE_ONLINE_COMMU:
    #     nowRecv = pool.http_server.total_bytes_recv
    #     #FeatSelectBytes = nowRecv-lastRecv
    #     print(
    #         "\n*********Online BitDecompose communication cost's: ",
    #         nowRecv - lastRecv,
    #         "bytes!*********\n",
    #     )
    #     lastRecv = nowRecv
    # if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
    #     end_time = time.time()
    #     #FeatSelectTime = end_time-start_time
    #     print(
    #         "\n*********Online BitDecompose computation time's: ",
    #         end_time - start_time,
    #         "s!*********\n",
    #     )
    #     start_time = end_time
    # ### bit_decomposition >>> ###
    #
    # ### <<< 1. FeatSelect ###
    # if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
    #     start_time = time.time()
    #     #initial_start_time = start_time
    # if BENCHMARK_MEASURE_ONLINE_COMMU:
    #     lastRecv = pool.http_server.total_bytes_recv
    #
    #
    # FaetSelectBeaverMat = dealer.distributeFeatSelect_BeaverTriple()
    # player.inputmat_XY(player.selectMatShare, player.sharesVec2Mat)
    # player.inputmat_beavers(FaetSelectBeaverMat[_id])
    # print("----------------------------")
    # print(player.mat_x_share)
    # print(player.mat_y_share)
    # print(player.mat_beavers[0])
    # print(player.mat_beavers[1])
    # print(player.mat_beavers[2])
    # print("----------------------------")
    # player.mat_mul()
    # if _id == 0:
    #     message = await pool.recv("P1")
    # else:
    #     message = await pool.recv("P0")
    # ef = pickle.loads(message)
    # mat_e = matrix_addition(player.mat_eshare ,ef[0])
    # print("mat_e:",mat_e)
    # mat_f = matrix_addition(player.mat_fshare,ef[1])
    # print("mat_f:",mat_f)
    # if _id == 0:
    #     tmp1 = matrix_multiply(player.mat_beavers[0],mat_f)
    #     tmp2 = matrix_multiply(mat_e, player.mat_beavers[1])
    #     tmp3 = matrix_addition(tmp1, tmp2)
    #     mat_zshare = matrix_addition(tmp3, player.mat_beavers[2])
    # else:
    #     tmp0 = matrix_multiply(mat_e, mat_f)
    #     tmp1 = matrix_multiply(player.mat_beavers[0] ,mat_f)
    #     tmp2 = matrix_multiply(mat_e, player.mat_beavers[1])
    #     tmp3 = matrix_addition(tmp0, tmp1)
    #     tmp4 = matrix_addition(tmp2, player.mat_beavers[2])
    #     mat_zshare = matrix_addition(tmp3, tmp4)
    # # 使用列表推导式展平矩阵
    # xvector = [item for sublist in mat_zshare for item in sublist]
    # player.inputxvector(xvector)
    # print("xvector:", xvector)
    # if BENCHMARK_MEASURE_ONLINE_COMMU:
    #     nowRecv = pool.http_server.total_bytes_recv
    #     FeatSelectBytes = nowRecv-lastRecv
    #     print(
    #         "\n*********Online FeatSelect communication cost's: ",
    #         nowRecv - lastRecv,
    #         "bytes!*********\n",
    #     )
    #     lastRecv = nowRecv
    # if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
    #     end_time = time.time()
    #     FeatSelectTime = end_time-start_time
    #     print(
    #         "\n*********Online FeatSelect computation time's: ",
    #         end_time - start_time,
    #         "s!*********\n",
    #     )
    #     start_time = end_time
    # ### 1.FeatSelect >>> ###
    #
    # ### <<< 2.NodeEval ###
    # if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
    #     start_time = time.time()
    # if BENCHMARK_MEASURE_ONLINE_COMMU:
    #     lastRecv = pool.http_server.total_bytes_recv
    # # 实现将十进制秘密共享 转换为二进制表示秘密共享
    # player.getdiff()
    # #x = [-3,1]
    # b = []
    # for ii in range(len(player.xvector)):
    #     player.inputbit_decomposition(player.diff[ii])
    #     player.inputbool_XY(player.a[0], player.b[0])
    #     player.bool_mul()
    #     if _id == 0:
    #         message = await pool.recv("P1")
    #     else:
    #         message = await pool.recv("P0")
    #     ef = pickle.loads(message)
    #     bool_e = player.bool_e_share^ef[0]
    #     bool_f = player.bool_f_share^ef[1]
    #     bool_z_share = (bool_e * bool_f * _id) ^ (bool_f * player.bool_beavers[0]) ^ \
    #             (bool_e * player.bool_beavers[1]) ^ player.bool_beavers[2]
    #     player.c[0] = bool_z_share
    #     player.x[0] = player.y[0]
    #     for i in range(1, BINARY_REPRESENTED):
    #         # a) [d] = [a]*[b]+1
    #         player.inputbool_XY(player.a[i], player.b[i])
    #         player.bool_mul()
    #         if _id == 0:
    #             message = await pool.recv("P1")
    #         else:
    #             message = await pool.recv("P0")
    #         ef = pickle.loads(message)
    #         bool_e = player.bool_e_share ^ ef[0]
    #         bool_f = player.bool_f_share ^ ef[1]
    #         bool_z_share = (bool_e * bool_f * _id) ^ (bool_f * player.bool_beavers[0]) ^ \
    #                        (bool_e * player.bool_beavers[1]) ^ player.bool_beavers[2]
    #         if _id == 0:
    #             player.d[i] = bool_z_share^0
    #         else:
    #             player.d[i] = bool_z_share^1
    #
    #         # b) [e] = [y]*[c_(i-1)]+1
    #         player.inputbool_XY(player.y[i], player.c[i-1])
    #         player.bool_mul()
    #         if _id == 0:
    #             message = await pool.recv("P1")
    #         else:
    #             message = await pool.recv("P0")
    #         ef = pickle.loads(message)
    #         bool_e = player.bool_e_share ^ ef[0]
    #         bool_f = player.bool_f_share ^ ef[1]
    #         bool_z_share = (bool_e * bool_f * _id) ^ (bool_f * player.bool_beavers[0]) ^ \
    #                        (bool_e * player.bool_beavers[1]) ^ player.bool_beavers[2]
    #         if _id == 0:
    #             player.e[i] = bool_z_share^0
    #         else:
    #             player.e[i] = bool_z_share^1
    #
    #         # c) [c] = [e]*[d]+1
    #         player.inputbool_XY(player.e[i], player.d[i])
    #         player.bool_mul()
    #         if _id == 0:
    #             message = await pool.recv("P1")
    #         else:
    #             message = await pool.recv("P0")
    #         ef = pickle.loads(message)
    #         bool_e = player.bool_e_share ^ ef[0]
    #         bool_f = player.bool_f_share ^ ef[1]
    #         bool_z_share = (bool_e * bool_f * _id) ^ (bool_f * player.bool_beavers[0]) ^ \
    #                        (bool_e * player.bool_beavers[1]) ^ player.bool_beavers[2]
    #         if _id == 0:
    #             player.c[i] = bool_z_share^0
    #         else:
    #             player.c[i] = bool_z_share^1
    #
    #         # d) [x] = [y]+[c_(i-1)]
    #         player.x[i] = player.y[i] ^ player.c[i - 1]
    #     print("player.x:",player.x[-1])
    #     b.append(player.x[-1])
    # player.inputcompare_res(b)
    # print("b list:",b)
    # #player.inputEvalRes(b)
    # if BENCHMARK_MEASURE_ONLINE_COMMU:
    #     nowRecv = pool.http_server.total_bytes_recv
    #     NodeEvalBytes = nowRecv-lastRecv
    #     print(
    #         "\n*********Online NodeEval communication cost's: ",
    #         nowRecv - lastRecv,
    #         "bytes!*********\n",
    #     )
    #     lastRecv = nowRecv
    # if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
    #     end_time = time.time()
    #     NodeEvalTime = end_time-start_time
    #     print(
    #         "\n*********Online NodeEval computation time's: ",
    #         end_time - start_time,
    #         "s!*********\n",
    #     )
    #     start_time = end_time
    # ### 2. NodeEval >>> ###
    #
    # ### <<< 3.GenRes ###
    # player.inputPath(pathpub)
    # if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
    #     start_time = time.time()
    #     #initial_start_time = start_time
    # if BENCHMARK_MEASURE_ONLINE_COMMU:
    #     lastRecv = pool.http_server.total_bytes_recv
    #
    # player.genres_pathcost()
    # pc = []
    # for i in range(len(player.pathPub)):
    #     tmp_path_cost = 0
    #     for j in range(len(player.pathcosts[0])):
    #         tmp_path_cost += player.pathcosts[i][j]
    #     pc.append(tmp_path_cost)
    # player.inputpc(pc)
    # player.gen_eval_res()
    # if _id == 0:
    #     message = await pool.recv("P1")
    # else:
    #     message = await pool.recv("P0")
    # pc_v_star = pickle.loads(message)
    # client_pc = [x + y for x, y in zip(player.pc_star, pc_v_star[0])]
    # print("client_pc:",client_pc)
    # client_v = [(x + y)%pow(2,BITS_REPRESENTED) for x, y in zip(player.v_star, pc_v_star[1])]
    # print("client_v:",client_v)
    # # print(pc)
    # # print("leafShares:", leafShares)
    # if BENCHMARK_MEASURE_ONLINE_COMMU:
    #     nowRecv = pool.http_server.total_bytes_recv
    #     GenResBytes = nowRecv-lastRecv
    #     print(
    #         "\n*********Online GenRes communication cost's: ",
    #         nowRecv - lastRecv,
    #         "bytes!*********\n",
    #     )
    #     lastRecv = nowRecv
    # if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
    #     end_time = time.time()
    #     GenResTime = end_time-start_time
    #     print(
    #         "\n*********Online GenRes computation time's: ",
    #         end_time - start_time,
    #         "s!*********\n",
    #     )
    #     start_time = end_time
    # ### 3.GenRes >>> ###
    #
    #
    # print("\n*********Online FeatSelect communication cost's: ",FeatSelectBytes,"bytes!*********\n",)
    # print("\n*********Online NodeEval communication cost's: ", NodeEvalBytes, "bytes!*********\n",)
    # print("\n*********Online GenRes communication cost's: ", GenResBytes, "bytes!*********\n", )
    #
    # print("\n*********Online FeatSelect computation time's: ",FeatSelectTime,"s!*********\n")
    # print("\n*********Online NodeEval computation time's: ",NodeEvalTime,"s!*********\n",)
    # print("\n*********Online GenRes compution time's: ", GenResTime, "s!*********\n", )
    # print("\n*********Total online computation time's: ",end_time - initial_start_time,"s!*********\n",)
    #
    #
    #
    #
    #
    # if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
    #     start_time = time.time()
    #     initial_start_time = start_time
    # if BENCHMARK_MEASURE_ONLINE_COMMU:
    #     lastRecv = pool.http_server.total_bytes_recv
    #
    # if _id == 0:
    #     message = await pool.recv("P1")
    #     await pool.send("P1", "Hello!，I am P0")
    #     print(message)
    # elif _id == 1:
    #     await pool.send("P0", "Hello!，I am P1 ")
    #     message = await pool.recv("P0")
    #     print(message)
    #
    # if BENCHMARK_MEASURE_ONLINE_COMMU:
    #     nowRecv = pool.http_server.total_bytes_recv
    #     #FeatSelectBytes = nowRecv-lastRecv
    #     print(
    #         "\n*********Online Test communication cost's: ",
    #         nowRecv - lastRecv,
    #         "bytes!*********\n",
    #     )
    #     lastRecv = nowRecv
    # if BENCHMARK_MEASURE_ONLINE_COMPUTATION:
    #     end_time = time.time()
    #     #FeatSelectTime = end_time-start_time
    #     print(
    #         "\n*********Online Test computation time's: ",
    #         end_time - start_time,
    #         "s!*********\n",
    #     )
    #     start_time = end_time
    await pool.shutdown()



if __name__ == "__main__":
    _id = int(sys.argv[1])
    #_id = 0
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_main(_id))
