class Extractor():
    '''Provides functions to extract features from a URL and its HTML code'''

    def __init__(self):
        pass

    def extract_features(self, url, html_code):
        '''Extracts the features from the given URL and HTML code and returns them as a dictionary'''

        # TODO: Call the different extraction functions
        features_dict = {
            "subdomain_level": 0,
            "url_length": 0,
            "num_dash_in_hostname": 0,
            "tilde_symbol": 0,
            "num_percent": 0,
            "num_ampersand": 0,
            "num_numeric_chars": 0,
            "domain_in_subdomains": 0,
            "https_in_hostname": 0,
            "path_length": 0,
            "double_slash_in_path": 0,
            "pct_ext_resource_urls": 0,
            "insecure_forms": 0,
            "ext_form_action": 0,
            "popup_window": 0,
            "iframe_or_frame": 0,
            "images_only_in_forms": 0,
            "abnormal_ext_form_action": 0,
        }
        return features_dict
    
    # ------------------------- Helper functions
    def get_subdomains(self, url):
        '''Returns a list of subdomains (namaes separated by dots in front of the domain) of the given URL'''
        return url.split("://")[1].split("/")[0].split(".")[:-2]
    
    def get_path(self, url):
        '''Returns the path (characters between hostnamen and a file: <protocol>://<host>/<path>/<file>?<query>) of the given URL'''
        return url.split("://")[1].split("/")[1].split("?")[0]

    def get_hostname(self, url):
        '''Returns the hostname of the given URL'''
        return url.split("://")[1].split("/")[0]

    def get_domain(self, url):
        '''Returns the domain of the given URL'''
        return url.split("://")[1].split("/")[0].split(".")[-2]
    
    # ------------------------- Extraction functions

    def extract_subdomain_level(self, url):
        pass

    def extract_url_length(self, url):
        pass

    def extract_num_dash_in_hostname(self, url):
        pass

    def extract_tilde_symbol(self, url):
        pass

    def extract_num_percent(self, url):
        pass

    def extract_num_ampersand(self, url):
        pass

    def extract_num_numeric_chars(self, url):
        pass

    def extract_domain_in_subdomains(self, url):
        pass

    def extract_https_in_hostname(self, url):
        pass

    def extract_path_length(self, url):
        pass

    def extract_double_slash_in_path(self, url):
        pass

    def extract_pct_ext_resource_urls(self, url):
        pass

    def extract_insecure_forms(self, url):
        pass

    def extract_ext_form_action(self, url):
        pass

    def extract_popup_window(self, url):
        pass

    def extract_iframe_or_frame(self, url):
        pass

    def extract_images_only_in_forms(self, url):
        pass

    def extract_abnormal_ext_form_action(self, url):
        pass

# Test the extractor
test_url = "https://www.google.com/search?q=hello&oq=hello&aqs=chrome..69i57j0l7.1002j0j7&sourceid=chrome&ie=UTF-8"
ex = Extractor()
# Test the helper functions
assert ex.get_subdomains(test_url) == ["www"]
assert ex.get_path(test_url) == "search"
assert ex.get_hostname(test_url) == "www.google.com"
assert ex.get_domain(test_url) == "google"
print("Helper functions work as expected")

# Test the extraction functions

