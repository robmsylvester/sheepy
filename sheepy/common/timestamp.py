from datetime import datetime


def get_timestamp_now():
    dateTimeObj = datetime.now(tz=datetime.now().astimezone().tzinfo)
    return dateTimeObj.strftime("%Y_%m_%d__%H_%M_%Z")
