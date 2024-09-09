sudo apt install -y python3-pip
sudo pip3 install -U pip gdown
sudo cp ./gdrive_model_repo_cookies.json $HOME/.cache/gdown/cookies.json
# sudo cp ./gdrive_model_repo_cookies.txt $HOME/.cache/gdown/cookies.txt
gdown --no-cookies --folder https://drive.google.com/drive/folders/1hoIl2-0ToMvZxXW7UGvCAvXpdQgh28Ms -O ${HOME}/model_repo