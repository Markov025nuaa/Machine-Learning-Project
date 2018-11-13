How we encode and decode image.

For date processing, we apply encoding and decoding to make a mask to the original image to distinguish the background sea and ship.

In order to encode connected refions as separeated masks, we do the following two steps.

First, we import skimage.morphology.label library and use built in **label** function to get an integer array of pixels, where all connected pixels are assigned the same integer value. Then we transform the integer array into a 0/1 formatted array, where 0 represents background(sea) while 1 represents foreground(ship).

Second, flattern the pixels array, then iterate each two pixels to calculate whether the values  are the same. Mark the index of the former pixel if two pixel values are different, and then put these indexes into a list. Afterwards, iterate the list in pairs and for each pair of numbers such as (A, B), change it into (A, B-A) . Now, we obtain a list which represents in a start-length format, where start is the start index of the pixel, length is the number of pixels with the same value of the start index of pixel.



Decode is just the opposite way of encoding. Get the list in start-length format and transform it to a 0/1 list, where 0 represents background while 1 represents foreground. Reshape this transformed list and set color to each integer.



![image-20181112191650658](/Users/wangchunxi/Library/Application Support/typora-user-images/image-20181112191650658.png)

To make sure our function can work, we need to check. Since check RLEs is more tedious and depending on how objects have been labeled might have different counts, we choose to check Images. The result is great that $Image_0$ is the same as $Image_1$. 

![image-20181112193543787](/Users/wangchunxi/Library/Application Support/typora-user-images/image-20181112193543787.png)





