from urllib.parse import urlparse
from bs4 import BeautifulSoup
import tldextract
import re
import socket

class Extractor():
    '''Provides functions to extract features from a URL and its HTML code'''

    def __init__(self):
        self.tlds_filepath = './services/tlds-alpha-by-domain.txt'

    def extract_features(self, url, html_code):
        '''Extracts the features from the given URL and HTML code and returns them as a dictionary'''

        features_dict = {}

        features_dict["NumDots"] = self.extract_num_dots(url)
        features_dict["SubdomainLevel"] = self.extract_subdomain_level(url)
        features_dict["PathLevel"] = self.extract_path_level(url)
        features_dict["UrlLength"] = self.extract_url_length(url)
        features_dict["NumDash"] = self.extract_num_dash(url)
        features_dict["NumDashInHostname"] = self.extract_num_dash_in_hostname(url)
        features_dict["AtSymbol"] = self.extract_at_symbol(url)
        features_dict["TildeSymbol"] = self.extract_tilde_symbol(url)
        features_dict["NumUnderscore"] = self.extract_num_underscore(url)
        features_dict["NumPercent"] = self.extract_num_percent(url)
        features_dict["NumAmpersand"] = self.extract_num_ampersand(url)
        features_dict["NumHash"] = self.extract_num_hash(url)
        features_dict["NumNumericChars"] = self.extract_num_numeric_chars(url)
        features_dict["IpAddress"] = self.extract_ip_address(url)
        features_dict["DomainInPaths"] = self.extract_domain_in_paths(url, self.tlds_filepath)
        features_dict["HttpsInHostname"] = self.extract_https_in_hostname(url)
        features_dict["HostnameLength"] = self.extract_hostname_length(url)
        features_dict["PathLength"] = self.extract_path_length(url)
        features_dict["QueryLength"] = self.extract_query_length(url)
        features_dict["DoubleSlashInPath"] = self.extract_double_slash_in_path(url)
        features_dict["NumSensitiveWords"] = self.extract_num_sensitive_words(url)
        features_dict["InsecureForms"] = self.extract_insecure_forms(html_code)
        features_dict["ExtFormAction"] = self.extract_ext_form_action(url, html_code)
        features_dict["PopupWindow"] = self.extract_popup_window(html_code)
        features_dict["SubmitInfoToEmail"] = self.extract_submit_info_to_email(html_code)
        features_dict["IframeOrFrame"] = self.extract_iframe_or_frame(html_code)
        features_dict["MissingTitle"] = self.extract_missing_title(html_code)
        features_dict["ImagesOnlyInForms"] = self.extract_images_only_in_forms(html_code)

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

    def get_tlds_from_file(self, file_path):
        '''Returns TLDs from official file'''
        with open(file_path, 'r') as file:
            tlds = [line.strip() for line in file]
        return tlds

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

    def extract_num_dots(self, url):
        '''Extracts the total number of dots of the URL and returns as an integer'''
        return url.count('.')
    
    def extract_path_level(self, url):
        '''Extracts path level of the URL'''
        parsed_url = urlparse(url)
        path_segments = parsed_url.path.strip('/').split('/')
        depth = len(path_segments)
        return depth
    
    def extract_num_dash(self, url):
        '''Extracts the total number of dashes in the URL'''
        return url.count('-')
    
    def extract_at_symbol(self, url):
        '''Checks if “@” symbol exist in webpage URL'''
        return '@' in url
    
    def extract_num_underscore(self, url):
        '''Counts the number of “_” in webpage URL'''
        return url.count('_')
    
    def extract_num_query_components(self, url):
        '''Counts the number of query parts in webpage URL'''
        pass
    
    def extract_num_hash(self, url):
        '''Counts the number of “#” in webpage URL'''
        return url.count("#")
    
    '''
    def extract_no_https(self, url):
        #Checks if HTTPS exist in webpage URL
        #An der Stelle schwierig zu sagen, ob generell der string https gemeint ist oder wirklich das protokoll
        #ich würde es zur Sicherheit weglassen
        return 'https' in url
    '''
    
    def extract_ip_address(self, url):
        '''Checks if IP address is used in hostname part of webpage URL'''
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
    
        if not hostname:
            return "Invalid"

        try:
            socket.inet_aton(hostname)
            return True
        except socket.error:
            return False

    def extract_domain_in_paths(self, url, file_path):
        '''Checks if TLD or ccTLD is used in the path of webpage URL'''
        parsed_url = urlparse(url)
        path = parsed_url.path

        last_path_part = path.split("/")[-1]

        tld_list = self.get_tlds_from_file(file_path)

        for tld in tld_list:
            if tld in last_path_part:
                return True

        return False

    def extract_hostname_length(self, url):
        '''Counts the total characters in hostname part of webpage URL'''
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname

        if hostname:
            return len(hostname)
        else:
            return 0

    def extract_query_length(self, url):
        '''Counts the total characters in query part of webpage URL'''
        parsed_url = urlparse(url)
        query = parsed_url.query

        return len(query)

    def extract_num_sensitive_words(self, url):
        '''Counts the number of sensitive words (i.e., “secure”, “account”, “webscr”, “login”,
        “ebayisapi”, “signin”, “banking”, “confirm”) in webpage URL'''
        sensitive_words = ["secure", "account", "webscr", "login", "ebayisapi", "signin", "banking", "confirm"]

        url_lower = url.lower()

        count = 0
        for word in sensitive_words:
            if word in url_lower:
                count += 1

        return count

    def extract_pct_ext_hyperlinks(self, html_code):
        pass

    def extract_ext_favicon(self, html_code):
        pass

    def extract_relative_form_action(self, html_code):
        pass

    def extract_abnormal_form_action(self, html_code):
        '''TO DO!!!'''
        pass

    def extract_frequent_domain_name_mismatch(self, url, html_code):
        pass

    def extract_right_click_disabled(self, html_code):
        pass

    def extract_submit_info_to_email(self, html_code):
        '''Check if HTML source code contains the HTML “mailto” function'''
        html_lower = html_code.lower()
        return "mailto:" in html_lower

    def extract_missing_title(self, html_code):
        '''Checks if the title tag is empty in HTML source code'''
        # Define a regular expression to match the title tag
        title_pattern = re.compile(r'<title>(.*?)</title>', re.DOTALL | re.IGNORECASE)

        # Search for the title tag in the HTML content
        match = title_pattern.search(html_code)

        # If the title tag is found, check if it's empty
        if match:
            title_content = match.group(1).strip()
            return title_content == ""
        else:
            return True

    def extract_subdomain_level_rt(self):
        pass

    def extract_pct_ext_resources_urls_rt(self, html_code):
        pass

    def extract_ext_meta_script_link_rt(self):
        pass

def test_extractor():
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
    assert ex.extract_num_dots(test_url) == 5
    assert ex.extract_path_level(test_url) == 1
    assert ex.extract_num_dash(test_url) == 1
    assert ex.extract_at_symbol(test_url) == False
    assert ex.extract_num_underscore(test_url) == 0
    assert ex.extract_num_hash(test_url) == 0
    #assert ex.extract_no_https(test_url) == True
    assert ex.extract_ip_address(test_url) == False
    assert ex.extract_domain_in_paths(test_url, ex.tlds_filepath) == False
    assert ex.extract_hostname_length(test_url) == 14
    assert ex.extract_query_length(test_url) == 72
    assert ex.extract_num_sensitive_words(test_url) == 0

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
    assert ex.extract_num_dots(test_url) == 6
    assert ex.extract_path_level(test_url) == 3
    assert ex.extract_num_dash(test_url) == 2
    assert ex.extract_at_symbol(test_url) == False
    assert ex.extract_num_underscore(test_url) == 0
    assert ex.extract_num_hash(test_url) == 0
    #assert ex.extract_no_https(test_url) == False
    assert ex.extract_ip_address(test_url) == False
    assert ex.extract_domain_in_paths(test_url, ex.tlds_filepath) == False
    assert ex.extract_hostname_length(test_url) == 27
    assert ex.extract_query_length(test_url) == 74
    assert ex.extract_num_sensitive_words(test_url) == 1

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
            <form action="http://www.google.com/search">
            <img src="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png">
            <input type="image" src="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png">    
            </form>
            <form action="https://www.google.com/search">
                <img src="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png">
                <img src="https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png">    
            </form>
            <frame src="https://www.google.com/search">
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

    # Test the complete extraction
    features_dict = ex.extract_features(test_url, test_html_code)
    print(features_dict)
    assert features_dict["SubdomainLevel"] == 2
    assert features_dict["UrlLength"] == 123
    assert features_dict["NumDashInHostname"] == 1
    assert features_dict["TildeSymbol"] == True
    assert features_dict["NumPercent"] == 1
    assert features_dict["NumAmpersand"] == 4
    assert features_dict["NumHash"] == 0
    assert features_dict["NumNumericChars"] == 13
    assert features_dict["IpAddress"] == False
    assert features_dict["DomainInPaths"] == False
    assert features_dict["HttpsInHostname"] == True
    assert features_dict["HostnameLength"] == 27
    assert features_dict["PathLength"] == 13
    assert features_dict["QueryLength"] == 74
    assert features_dict["DoubleSlashInPath"] == True
    assert features_dict["NumSensitiveWords"] == 1
    assert features_dict["InsecureForms"] == True
    assert features_dict["ExtFormAction"] == True
    assert features_dict["PopupWindow"] == True
    assert features_dict["IframeOrFrame"] == True
    assert features_dict["ImagesOnlyInForms"] == True
    assert features_dict["NumDots"] == 6
    assert features_dict["PathLevel"] == 3
    assert features_dict["NumDash"] == 2
    assert features_dict["AtSymbol"] == False
    assert features_dict["NumUnderscore"] == 0
    assert features_dict["SubmitInfoToEmail"] == False
    assert features_dict["MissingTitle"] == False
    print("Complete extraction works as expected")

if __name__ == "__main__":
    test_extractor()
