###Contributor: Hang Yu

```Please``` place dataset in this folder

---

```chmod +x download.sh;./download.sh```

It will take a few minutes, the whole data set is about 900MB unzipped under the ```smallNORB``` folder.

The TFRecord generated is about 3.6GB each train and test dataset. But you can write only a proportion by modifying        ```smallNORB.py```. Don't worry, it's fast as tensorflow api employs multithreading.

Under ```data``` folder, type ```python smallNORB.py tfrecord```. You will see multiple tfrecords with extension ```tfrecord```. Follow the instructions in ```test``` method in ```smallNORB.py``` to read files and parse them into batches
