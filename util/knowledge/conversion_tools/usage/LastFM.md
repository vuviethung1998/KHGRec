# Last.FM

1.**Clone the repository and install requirements.** 
(If you have already done this, please move to the step 2.)

```
git clone https://github.com/RUCAIBox/RecDatasets

cd RecDatasets/conversion_tools

pip install -r requirements.txt
```

2.**Download the Last.FM Dataset and extract the dataset file.**
(If you have already done this, please move to the step 3.)

```
wget http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

unzip hetrec2011-lastfm-2k.zip -d ./lastfm
```

3.**Go the ``conversion_tools/`` directory 
and run the following command to get the atomic files of Last.FM dataset.**

```
python run.py --dataset lastfm \
--input_path lastfm --output_path output_data/lastfm \
--convert_inter --convert_item
```

`input_path` is the path of the input decompressed LastFM file

`output_path` is the path to store converted atomic files

 `convert_inter` Last.FM can be converted to '*.inter' atomic file

`convert_item` Last.FM can be converted to '*.item' atomic file

4. **Add knowledge**
```
python add_knowledge.py --dataset lastfm --inter_file output_data/lastfm/lastfm.inter --kg_data_path ../dataset/lastfm/LFM-1b-KG/LFM-1b-KG --output_path output_data/lastfm/ --hop=2
```