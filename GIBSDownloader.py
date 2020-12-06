import requests
import enum


class ZoomLevel(enum.Enum):
    FOUR = 4
    EIGHT = 8


class GIBSDownloader:

    def __init__(self, zoom, date, folder):
        self.zoom_level = zoom  #has to be string FOUR or EIGHT
        self.date = date  #yyyy-mm-dd
        self.folder = folder

    def download(self):

        if(self.zoom_level == ZoomLevel.EIGHT):
            counter = 0
            for i in range(0, 160):
                for j in range(0, 320):

                    with open(self.folder+str(counter)+'.jpg', 'wb') as handle:
                        response = requests.get("https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/"+self.date+"/250m/"+self.zoom_level+"/"+str(i)+"/"+str(j)+".jpg", stream=True)

                        if not response.ok:
                            print(response)

                        for block in response.iter_content(1024):
                            if not block:
                                break

                        handle.write(block)
                        print(str(i), str(j))

                        counter += 1
            if(self.zoom_level == ZoomLevel.FOUR):
                counter = 0
                for i in range(0, 10):
                    for j in range(0, 20):
                        with open(self.folder+str(counter)+'.jpg', 'wb') as handle:
                            response = requests.get("https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/"+self.date+"/250m/"+self.zoom_level+"/"+str(i)+"/"+str(j)+".jpg",stream=True)

                            if not response.ok:
                                print(response)

                            for block in response.iter_content(1024):
                                if not block:
                                    break

                                handle.write(block)
                        print(str(i), str(j))

                        counter += 1
