###Contributor: Hang Yu

```Please``` place dataset in this folder

---

```chmod +x download.sh;./download.sh```

It will take a few minutes, the whole data set is about 900MB unzipped.

The TFRecord generated is about 3.6GB each train and test dataset. But you can write only a proportion by modifying        ```dataset.py```. Don't worry, it's fast as tensorflow api employs multithreading.

GO back to root folder
```python dataset.py tfrecord```
You will see multiple tfrecords under ```./data```
