###Contributor: Hang Yu

---

```chmod +x download.sh;./download.sh```

Download will take a few minutes or seconds, the whole data set is about 900MB unzipped. They are placed under the ```smallNORB``` folder.

We will generate TFRecord about 3.6GB for both train and test dataset. Tensorflow api employs multithreading, so this process would be fast (within a minute).

~~Under ```data``` folder, type ```python smallNORB.py tfrecord```. You will see multiple tfrecords with extension ```tfrecord```. Follow the instructions in ```test``` method in ```smallNORB.py``` to read files and parse them into batches.~~
