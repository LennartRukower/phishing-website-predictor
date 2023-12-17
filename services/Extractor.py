from urllib.parse import urlparse
import re

class Extractor():
    '''Provides functions to extract features from a URL and its HTML code'''

    def __init__(self):
        pass

    def extract_features(self, url, html_code):
        '''Extracts the features from the given URL and HTML code and returns them as a dictionary'''

        features_dict = {}

        features_dict["SubdomainLevel"] = self.extract_subdomain_level(url)
        features_dict["UrlLength"] = self.extract_url_length(url)
        features_dict["NumDashInHostname"] = self.extract_num_dash_in_hostname(url)
        features_dict["TildeSymbol"] = self.extract_tilde_symbol(url)
        features_dict["NumPercent"] = self.extract_num_percent(url)
        features_dict["NumAmpersand"] = self.extract_num_ampersand(url)
        features_dict["NumNumericChars"] = self.extract_num_numeric_chars(url)
        features_dict["HttpsInHostname"] = self.extract_https_in_hostname(url)
        features_dict["PathLength"] = self.extract_path_length(url)
        features_dict["DoubleSlashInPath"] = self.extract_double_slash_in_path(url)
        features_dict["InsecureForms"] = self.extract_insecure_forms(html_code)
        features_dict["ExtFormAction"] = self.extract_ext_form_action(url, html_code)
        features_dict["PopupWindow"] = self.extract_popup_window(html_code)
        features_dict["IframeOrFrame"] = self.extract_iframe_or_frame(html_code)
        features_dict["ImagesOnlyInForms"] = self.extract_images_only_in_forms(html_code)
        features_dict["NumDots"] = self.extract_num_dots(url)
        features_dict["PathLevel"] = self.extract_path_level(url)
        features_dict["NumDash"] = self.extract_num_dash(url)
        features_dict["AtSymbol"] = self.extract_at_symbol(url)
        features_dict["NumUnderscore"] = self.extract_num_underscore(url)
        features_dict["NumHash"] = self.extract_num_hash(url)
        features_dict["IpAddress"] = self.extract_ip_address(url)
        features_dict["DomainInPaths"] = self.extract_domain_in_paths(url, self.tlds_filepath)
        features_dict["HostnameLength"] = self.extract_hostname_length(url)
        features_dict["QueryLength"] = self.extract_query_length(url)
        features_dict["NumSensitiveWords"] = self.extract_num_sensitive_words(url)
        features_dict["SubmitInfoToEmail"] = self.extract_submit_info_to_email(html_code)
        features_dict["MissingTitle"] = self.extract_missing_title(html_code)

        return features_dict
    
    # ------------------------- Helper functions
    def get_subdomains(self, url):
        '''Returns a list of subdomains (namaes separated by dots in front of the domain) of the given URL'''
        return url.split("://")[1].split("/")[0].split(".")[:-2]
    
    def get_path(self, url):
        '''Returns the path (characters between hostnamen and a file: <protocol>://<host>/<path>/<file>?<query>) of the given URL'''
        url_path = urlparse(url).path
        url_path = url_path[1:] # Remove first slash
        file_pattern = r'/[^/]+\.[^/]+$'
        if re.search(file_pattern, url_path):
            return re.sub(file_pattern, '', url_path)
        else:
            return url_path
        

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
assert ex.extract_subdomain_level(test_url) == 1
assert ex.extract_url_length(test_url) == 102
assert ex.extract_num_dash_in_hostname(test_url) == 0
assert ex.extract_tilde_symbol(test_url) == False
assert ex.extract_num_percent(test_url) == 0
assert ex.extract_num_ampersand(test_url) == 4
assert ex.extract_num_numeric_chars(test_url) == 13
assert ex.extract_https_in_hostname(test_url) == False
assert ex.extract_path_length(test_url) == 6
assert ex.extract_double_slash_in_path(test_url) == False

# Insecure and posibble phishing test url
test_url = "http://https.secure-url.googel.com/search//value?~q=hello&oq=hello&aqs=chrome..69i57j0l7.1002j0j7&sourceid=chrome&ie=UTF-8%"
assert ex.extract_subdomain_level(test_url) == 2
assert ex.extract_url_length(test_url) == 123
assert ex.extract_num_dash_in_hostname(test_url) == 1
assert ex.extract_tilde_symbol(test_url) == True
assert ex.extract_num_percent(test_url) == 1
assert ex.extract_num_ampersand(test_url) == 4
assert ex.extract_num_numeric_chars(test_url) == 13
assert ex.extract_https_in_hostname(test_url) == True
assert ex.extract_path_length(test_url) == 13
assert ex.extract_double_slash_in_path(test_url) == True

print("Extraction functions work as expected")

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
