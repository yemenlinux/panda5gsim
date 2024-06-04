from panda5gSim.core.los_probability import P_LoS_Rec_ITU_R_P_1410_5

def test_P_LoS_Rec_ITU_R_P_1410_5():
    # test the P_LoS_Rec_ITU_R_P_1410_5 function
    # test case 1
    distance2d = 0.15
    h_tx = 50
    h_rx = 1.5
    alpha = 0.5
    beta = 400
    gamma = 20
    # expected result
    expected_result = 0.17262622533505
    # actual result
    actual_result = P_LoS_Rec_ITU_R_P_1410_5(distance2d, h_tx, h_rx, alpha, beta, gamma)
    # compare the results
    assert expected_result == actual_result, f"The actual result is not the expected result."
     
