from urllib.parse import urlparse
from bs4 import BeautifulSoup
import tldextract
import re

class Extractor():
    '''Provides functions to extract features from a URL and its HTML code'''

    def __init__(self):
        pass

    def extract_features(self, url, html_code):
        '''Extracts the features from the given URL and HTML code and returns them as a dictionary'''

        # TODO: Call the different extraction functions
        features_dict = {
            "SubdomainLevel": 0,
            "UrlLength": 0,
            "NumDashInHostname": 0,
            "TildeSymbol": False,
            "NumPercent" : 0,
            "NumAmpersand" : 0,
            "NumNumericChars" : 0,
            "DomainInSubdomains" : False,
            "HttpsInHostname" : False,
            "PathLength" : 0,
            "DoubleSlashInPath" : False,
            "PctExtResourceUrls" : 0.0,
            "InsecureForms" : False,
            "ExtFormAction" : False,
            "PopUpWindow" : False,
            "IframeOrFrame" : False,
            "ImagesOnlyInForm" : False,
        }
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
    
    def get_forms(self, html_code):
        '''Returns a list of forms in the given HTML code'''
        soup = BeautifulSoup(html_code, 'html.parser')
        forms = soup.find_all('form')
        return forms

    # TODO
    def get_urls(self, html_code):
        raise NotImplementedError
        '''Returns a list of urls in the given HTML code'''
        soup = BeautifulSoup(html_code, 'html.parser')
        # Extract all hyperlinks from the HTML code
        hyperlinks = soup.find_all('a')
        # Extract the URLs from the hyperlinks
        urls = [link.get('href') for link in hyperlinks]
        return urls
    
    # TODO: Move to utils
    def get_main_domain(self, url):
        """
        Extracts the main domain from a URL using tldextract, which handles
        cases including second-level country code TLDs.

        PARAMETERS:
        ------
        url (str): The URL to extract the main domain from.

        RETURNS:
        ------
        str: The main domain of the URL.
        """
        extracted = tldextract.extract(url)
        # Combine the registered domain and the TLD
        main_domain = "{}.{}".format(extracted.domain, extracted.suffix)
        return main_domain
    
    def is_url_external(self, url_to_check, base_url):
        '''Returns whether the given URL is external (points to a different domain - ignore subdomains)'''
        base_domain = self.get_main_domain(base_url)
        check_domain = self.get_main_domain(url_to_check)
        print(base_domain, check_domain)

        return base_domain.lower() != check_domain.lower()


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
        urls = self.get_urls(html_code)
        pass

    def extract_insecure_forms(self, html_code):
        '''Extracts whether the given HTML code contains insecure forms (action attribute does not start with https) and returns it as a boolean'''
        forms = self.get_forms(html_code)
        # Get list of action attributes of the forms
        form_actions = [form.get('action') for form in forms]
        # Check if any of the form actions is insecure (does not start with https)
        for action in form_actions:
            if not action.startswith("https"):
                return True
        return False

    def extract_ext_form_action(self, url, html_code):
        forms = self.get_forms(html_code)
        # Get list of action attributes of the forms
        form_actions = [form.get('action') for form in forms]
        # Check if any of the form actions contains an external URL
        for action in form_actions:
            if self.is_url_external(action, url):
                return True
        return False

    def extract_popup_window(self, html_code):
        '''Extracts whether the given HTML code contains a popup window and returns it as a boolean'''
        soup = BeautifulSoup(html_code, 'html.parser')
        scripts = soup.find_all('script')

        # Patterns to search for in JavaScript
        popup_patterns = [r'window\.open\(', r'alert\(', r'confirm\(', r'prompt\(']

        for script in scripts:
            if script.string:
                for pattern in popup_patterns:
                    if re.search(pattern, script.string):
                        return True

        return False

    def extract_iframe_or_frame(self, html_code):
        '''Extracts whether the given HTML code contains an iframe or frame and returns it as a boolean'''
        soup = BeautifulSoup(html_code, 'html.parser')
        iframes = soup.find_all('iframe')
        frames = soup.find_all('frame')
        return len(iframes) > 0 or len(frames) > 0

    def extract_images_only_in_forms(self, html_code):
        '''Extracts whether the given HTML code contains forms that only contain images and returns it as a boolean'''
        soup = BeautifulSoup(html_code, 'html.parser')
        forms = soup.find_all('form')

        for form in forms:
            form_elements = form.find_all()
            # TODO: if all(elem.name == 'img' or (elem.name == 'input' and elem.get('type') == 'image') for elem in form_elements): 
            if all(elem.name == 'img' for elem in form_elements):
                return True

        return False


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
        <form action="https://www.evil.com/search">
            <input type="text" name="q">
            <input type="submit" value="Search">
        </form> 
        <form action="https://www.google.com/search">
        <img src="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png">
        <input type="image" src="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png">    
        </form>
        <form action="https://www.google.com/search">
            <img src="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png">
            <img src="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png">    
        </form>
        <a href="https://www.google.com/search">Search</a>
        <script>
            window.open("https://www.google.com/search");
        </script>
    </body>
</html>
"""
#print(ex.get_urls(test_html_code))
print(ex.extract_insecure_forms(test_html_code))
print(ex.extract_ext_form_action(test_url, test_html_code))
print(ex.extract_popup_window(test_html_code))
print(ex.extract_iframe_or_frame(test_html_code))
print(ex.extract_images_only_in_forms(test_html_code))