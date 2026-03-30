import torch
from ZeroDCE import ZeroDCE
from Retinex import Retinexformer
from GMFN import GeneralMethodFlowNetwork
from Fusion import Fusion
import gdown


device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")



#----------------------------------------------------------------------------------#
#            D O W N L O A D    P R E - T R A I N E D   W E I G H T S              #
#----------------------------------------------------------------------------------#

retinex_model_id = "1hheOLJGxswUgm7fM8A9gspDFyfNQiaYc"
gmfn_model_id = "1Rb_Yw45vTiph1i_wv6uLcNtyTjFJL95L"
zerodce_model_id = "1dxkcXGz8XfOeJKBoa1L--ZvkXkL3o6d4"
fusion_model_id = "1yP-SnN0Amo0flyewFTAXkrQtQ9miLbbi"

retinex_model_name = "model-retinex.pth"
gmfn_model_name = "model-gmfn.pth"
zerodce_model_name = "model-zero.pth"
fusion_model_name = "model-fusion.pth"



def download_weights(file_id, model_name):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(model_name):
        with st.spinner(f"Downloading model {model_name} weights from Google Drive..."):
            gdown.download(url, model_name, quiet=False)
    return model_name

#----------------------------------------------------------------------------------#
#            L O A D I N G    P R E - T R A I N E D   W E I G H T S                #
#----------------------------------------------------------------------------------#

def load_weights():
    download_weights(zerodce_model_id, zerodce_model_name)
    model_zero = ZeroDCE()
    state_dict = torch.load(zero_model_name, map_location=device)
    model_zero.load_state_dict(state_dict)
    
    download_weights(retinex_model_id, retinex_model_name)
    model_retinex = RetinexFormer()
    state_dict = torch.load(retinex_model_name, map_location=device)
    model_retinex.load_state_dict(state_dict)

    download_weights(gmfn_model_id, gmfn_model_name)
    model_gmfn = GeneralMethodFlowNetwork()
    state_dict = torch.load(gmfn_model_name, map_location=device)
    model_gmfn.load_state_dict(state_dict)
    
    download_weights(fusion_model_id, fusion_model_name)
    model_fusion = Fusion(model_zero, model_Ret, model_gmfn)
    state_dict = torch.load(fusion_model_name, map_location=device)
    model_fusion.load_state_dict(state_dict)


    return model_fusion
