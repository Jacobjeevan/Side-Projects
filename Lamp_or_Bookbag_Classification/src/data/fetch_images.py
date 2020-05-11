from requests import exceptions
import argparse
import requests
import cv2
import os
from pathlib import Path


class fetch_images():
    def __init__(self, args):
        '''Initialize the class; parse the arguments from command line (search query, api key and the output folder)/
        We will also initialize other parameters (GROUP SIZE, MAX_RESULTS etc) that will be used elsewhere'''
        self.subscription_key = args.key
        self.search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
        self.MAX_RESULTS = 300 #Total number of images to fetch
        self.GROUP_SIZE = 50 #Total # of images per page (in this case each page will contain 50 images; 300/50 = 6 pages will be scraped)
        #Set some possible exceptions likely to happen during scraping, so we can catch them.
        self.EXCEPTIONS = set([IOError, FileNotFoundError, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError, exceptions.Timeout])
        self.search_term = args.query
        self.output = args.output

    def getImages(self):
        #Reference on Bing API: https://docs.microsoft.com/en-us/azure/cognitive-services/bing-image-search/quickstarts/python
        #Most of the code below is from: https://www.pyimagesearch.com/2018/04/09/how-to-quickly-build-a-deep-learning-image-dataset/
        headers = {"Ocp-Apim-Subscription-Key" : self.subscription_key}
        #Check https://docs.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-images-api-v7-reference for other useful parameters
        params = {"q": self.search_term, "offset": 0, "count": self.GROUP_SIZE, "imageType": "photo"}
        print("[INFO] searching Bing API for '{}'".format(self.search_term))
        response = requests.get(self.search_url, headers=headers, params=params)
        response.raise_for_status()
        # grab the results from the search, including the total number of
        # estimated results returned by the Bing API
        results = response.json()
        estNumResults = min(results["totalEstimatedMatches"], self.MAX_RESULTS)
        print("[INFO] {} total results for '{}'".format(estNumResults,
            self.search_term))
        total = 0
        for offset in range(0, estNumResults, self.GROUP_SIZE):
            print("[INFO] making request for group {}-{} of {}...".format(
                offset, offset + self.GROUP_SIZE, estNumResults))
            params["offset"] = offset
            response = requests.get(self.search_url, headers=headers, params=params)
            response.raise_for_status()
            results = response.json()
            print("[INFO] saving images for group {}-{} of {}...".format(
                offset, offset + self.GROUP_SIZE, estNumResults))
            for v in results["value"]:
                try:
                    # Send a get request to download the image
                    print("[INFO] fetching: {}".format(v["contentUrl"]))
                    r = requests.get(v["contentUrl"], timeout=30)
                    # build the path to the output image
                    ext = v["contentUrl"][v["contentUrl"].rfind("."):]
                    p = os.path.sep.join([self.output, "{}{}".format(
                        str(total).zfill(8), ext)])
                    # Save the image to disk
                    f = open(p, "wb")
                    f.write(r.content)
                    f.close()
                # Catch errors
                except Exception as e:
                    # Check if it's an exception on the list, if so print message and skip
                    if type(e) in self.EXCEPTIONS:
                        print("[INFO] skipping: {}".format(v["contentUrl"]))
                        continue

                image = cv2.imread(p)
                # if the image is `None` then we could not properly load the
                # image from disk (so it should be ignored)
                if image is None:
                    print("[INFO] deleting: {}".format(p))
                    os.remove(p)
                    continue
                # update the counter
                total += 1


def build_parser():
    #Use build parser for setting commandline arguments (including default values)
    savepath = "../../data/raw"
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True,
	help="Search Query")
    parser.add_argument("-k", "--key", required=True,
	help="API Key")
    parser.add_argument("-o", "--output", default=savepath,
	help="Path to output directory")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    #Grabs the current directory
    dirname = os.path.dirname(__file__)
    #Builds the path to save location and creates a folder for the query term inside savepath.
    args.output = os.path.join(dirname, args.output, args.query)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    fetch_images(args).getImages()

if __name__ == "__main__":
    main()