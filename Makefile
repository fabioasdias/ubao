all:
	python3 preprocessImages.py
	python3 genProj.py 
	mv imglist.json ./ui/public/imglist.json
