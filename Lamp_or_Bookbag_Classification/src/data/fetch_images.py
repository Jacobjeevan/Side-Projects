from requests import exceptions
import argparse
import requests
import cv2
import os


def build_parser():
    savepath = "../../data/raw"
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True,
	help="Search Query")
    parser.add_argument("-k", "--key", required=True,
	help="API Key")
    parser.add_argument("-o", "--output", default=savepath,
	help="Path to output directory")
    return parser



# construct the argument parser and parse the arguments
parser = build_parser()
args = parser.parse_args()


subscription_key = args.key
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"

MAX_RESULTS = 300
GROUP_SIZE = 50

EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

search_term = args.query
headers = {"Ocp-Apim-Subscription-Key" : subscription_key}
params = {"q": search_term, "offset": 0, "count": GROUP_SIZE, "imageType": "photo"}
# make the search
print("[INFO] searching Bing API for '{}'".format(search_term))
response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
# grab the results from the search, including the total number of
# estimated results returned by the Bing API
results = response.json()
estNumResults = min(results["totalEstimatedMatches"], MAX_RESULTS)
print("[INFO] {} total results for '{}'".format(estNumResults,
	search_term))
# initialize the total number of images downloaded thus far
total = 0

for offset in range(0, estNumResults, GROUP_SIZE):
	print("[INFO] making request for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))
	params["offset"] = offset
	response = requests.get(search_url, headers=headers, params=params)
	response.raise_for_status()
	results = response.json()
	print("[INFO] saving images for group {}-{} of {}...".format(
		offset, offset + GROUP_SIZE, estNumResults))


for v in results["value"]:
    # try to download the image
    try:
        # make a request to download the image
        print("[INFO] fetching: {}".format(v["contentUrl"]))
        r = requests.get(v["contentUrl"], timeout=30)
        # build the path to the output image
        ext = v["contentUrl"][v["contentUrl"].rfind("."):]
        p = os.path.sep.join([args["output"], "{}{}".format(
            str(total).zfill(8), ext)])
        # write the image to disk
        f = open(p, "wb")
        f.write(r.content)
        f.close()
    # catch any errors that would not unable us to download the
    # image
    except Exception as e:
        # check to see if our exception is in our list of
        # exceptions to check for
        if type(e) in EXCEPTIONS:
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