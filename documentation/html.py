import dominate
from dominate.tags import meta, style, h3, table, th, tr, td, b, p, a, img, br
import os


class HTML:
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.web_dir = web_dir
        self.img_dir = os.path.join(self.web_dir, 'images')
        if not os.path.exists(self.web_dir):
            os.makedirs(self.web_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))
        with self.doc.head:
            style("table {"
                  "    border: 1px solid black;"
                  "    border-collapse: collapse;"
                  "}"
                  "th {"
                  "    border-bottom: 1px solid black;"
                  "    text-align: left;"
                  "}"
                  "th, td {"
                  "    padding-left:5px;"
                  "    padding-right:5px;"
                  "}")


    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file

        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file

        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """
        self.t = table(border=1, style="table-layout: fixed; margin: 10px;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=os.path.join('images', link)):
                                img(style="width:%dpx" % width, src=os.path.join('images', im))
                            br()
                            p(txt)

    def add_table(self, values, x_labels=None, y_labels=None):
        """add table to the HTML file

        Parameters:
            values (list of str lists) --  The content of the table. This should be a classic 2d array.
            x_labels (str list)   -- a list of labels for x-axis
            y_labels (str list)  -- a list of labels for y-axis
        """
        self.t = table(style="margin: 10px; padding:4px;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            if x_labels:
                with tr():
                    if y_labels:
                        th(style="border-right: 1px solid black;")
                    for label in x_labels:
                        with th():
                            b(label)
            for i in range(len(values)):
                with tr():
                    if y_labels:
                        with td(style="border-right: 1px solid black;"):
                            b(y_labels[i])
                    for content in values[i]:
                        td(content)

    def save(self):
        """save the current content to the HMTL file"""
        html_file = '%s/index.html' % self.web_dir
        f = open(html_file, 'wt')
        f.write(self.doc.render())
        f.close()


if __name__ == '__main__':  # we show an example usage here.
    html = HTML('web/', 'test_html')
    html.add_header('hello world')

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
