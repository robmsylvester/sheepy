import requests


def download_resource(key: str, output_path: str) -> str:
    """[summary]

    Args:
        s3_path (str): [description]
        output_path (str): [description]

    Raises:
        botocore.exceptions.ClientError - The file does not exist at the specified s3 path
    """
    url = "https://robmsylvester.s3.us-west-1.amazonaws.com/{}".format(key)
    headers = {"Host": "robmsylvester.s3.us-west-1.amazonaws.com"}
    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path
