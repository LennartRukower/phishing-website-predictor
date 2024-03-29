import requests

class WebCrawler:
    '''
    Provides a method to crawl a website and return the html code
    '''

    def __init__(self):
        pass

    def crawl(self, url):
        '''
        Crawls the website of the given `url` and returns the html code

        PARAMETERS
        ----------
        url : str
            The url of the website to crawl
        
        RETURNS
        -------
        html_code : str, url: str
        '''

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code
            return response.text, response.url
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"