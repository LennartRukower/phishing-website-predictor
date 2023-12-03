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
            "tilde_symbol": False,
            "num_percent": 0,
            "num_ampersand": 0,
            "num_numeric_chars": 0,
            "domain_in_subdomains": False,
            "https_in_hostname": False,
            "path_length": 0,
            "double_slash_in_path": False,
            "pct_ext_resource_urls": 0.0,
            "insecure_forms": False,
            "ext_form_action": False,
            "popup_window": False,
            "iframe_or_frame": False,
            "images_only_in_forms": False,
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
        '''Extracts the subdomain level of the given URL and returns it as an integer'''
        return len(self.get_subdomains(url))

    def extract_url_length(self, url):
        '''Extracts the length of the given URL and returns it as an integer'''
        return len(url)

    def extract_num_dash_in_hostname(self, url):
        '''Extracts the number of dashes in the hostname of the given URL and returns it as an integer'''
        return self.get_hostname(url).count("-")

    def extract_tilde_symbol(self, url):
        '''Extracts whether the tilde symbol is in the given URL and returns it as a boolean'''
        return "~" in url

    def extract_num_percent(self, url):
        '''Extracts the number of percent symbols in the given URL and returns it as an integer'''
        return url.count("%")

    def extract_num_ampersand(self, url):
        '''Extracts the number of ampersands in the given URL and returns it as an integer'''
        return url.count("&")

    def extract_num_numeric_chars(self, url):
        '''Extracts the number of numeric characters in the given URL and returns it as an integer'''
        return sum([1 for c in url if c.isdigit()])

    def extract_domain_in_subdomains(self, url):
        '''Extracts whether the TLDs or ccTLDs are in the subdomains of the given URL and returns it as a boolean'''
        raise NotImplementedError

    def extract_https_in_hostname(self, url):
        '''Extracts whether the string "https" is in the hostname of the given URL and returns it as a boolean'''
        return "https" in self.get_hostname(url)

    def extract_path_length(self, url):
        '''Extracts the length of the path of the given URL and returns it as an integer'''
        return len(self.get_path(url))

    def extract_double_slash_in_path(self, url):
        '''Extracts whether the string "//" is in the path of the given URL and returns it as a boolean'''
        return "//" in self.get_path(url)

    def extract_pct_ext_resource_urls(self, url, html_code):
        pass

    def extract_insecure_forms(self, html_code):
        pass

    def extract_ext_form_action(self, url, html_code):
        pass

    def extract_popup_window(self, html_code):
        pass

    def extract_iframe_or_frame(self, html_code):
        pass

    def extract_images_only_in_forms(self, ht):
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
test_html_code = """
<html>
    <head>
        <title>Test</title>
    </head>
    <body>
        <form action="https://www.google.com/search">
            <input type="text" name="q">
            <input type="submit" value="Search">
        </form>
        <script>
            window.open("https://www.google.com/search");
        </script>
    </body>
</html>
"""
