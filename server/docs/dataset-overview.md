# Features Information

| No. | Identifier                   | Value type  | Description                                                                                                                                               |
|-----|------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | NumDots                      | Discrete    | Counts the number of dots in webpage URL.                                                                                                        |
| 2   | **SubdomainLevel**               | Discrete    | Counts the level of subdomain in webpage URL.                                                                                                  |
| 3   | PathLevel                    | Discrete    | Counts the depth of the path in webpage URL.                                                                                                          |
| 4   | **UrlLength**                    | Discrete    | Counts the total characters in the webpage URL.                                                                                                |
| 5   | NumDash                      | Discrete    | Counts the number of “-” in webpage URL.                                                                                            |
| 6   | **NumDashInHostname**            | Discrete    | Counts the number of “-” in hostname part of webpage URL.                                                                           |
| 7   | AtSymbol                     | Binary      | Checks if “@” symbol exist in webpage URL.                                                                                    |
| 8   | **TildeSymbol**                  | Binary      | Checks if “ ∼ ” symbol exist in webpage URL.                                                                                                           |
| 9   | NumUnderscore                | Discrete    | Counts the number of “_” in webpage URL.                                                                                                               |
| 10  | **NumPercent**                   | Discrete    | Counts the number of “%” in webpage URL.                                                                                                               |
| 11  | NumQueryComponents           | Discrete    | Counts the number of query parts in webpage URL.                                                                                                       |
| 12  | **NumAmpersand**                 | Discrete    | Counts the number of “&” in webpage URL.                                                                                                               |
| 13  | NumHash                      | Discrete    | Counts the number of “#” in webpage URL.                                                                                                               |
| 14  | **NumNumericChars**              | Discrete    | Counts the number of numeric characters in the webpage URL.                                                                                            |
| 15  | NoHttps                      | Binary      | Checks if HTTPS exist in webpage URL.                                                                                          |
| 16  | RandomString                 | Binary      | Checks if random strings exist in webpage URL.                                                                                                         |
| 17  | IpAddress                    | Binary      | Checks if IP address is used in hostname part of webpage URL.                                                                 |
| 18  | **DomainInSubdomains**           | Binary      | Checks if TLD or ccTLD is used as part of subdomain in webpage URL.                                                                                   |
| 19  | DomainInPaths                | Binary      | Checks if TLD or ccTLD is used in the path of webpage URL.                                                                                |
| 20  | **HttpsInHostname**              | Binary      | Checks if HTTPS in obfuscated in hostname part of webpage URL.                                                                                             |
| 21  | HostnameLength               | Discrete    | Counts the total characters in hostname part of webpage URL.                                                                                          |
| 22  | **PathLength**                   | Discrete    | Counts the total characters in path of webpage URL.                                                                                                   |
| 23  | QueryLength                  | Discrete    | Counts the total characters in query part of webpage URL.                                                                                             |
| 24  | **DoubleSlashInPath**            | Binary      | Checks if “//” exist in the path of webpage URL.                                                                                                      |
| 25  | NumSensitiveWords            | Discrete    | Counts the number of sensitive words (i.e., “secure”, “account”, “webscr”, “login”, “ebayisapi”, “signin”, “banking”, “confirm”) in webpage URL. |
| 26  | EmbeddedBrandName            | Binary      | Checks if brand name appears in subdomains and path of webpage URL. Brand name here is assumed as the most frequent domain name in the webpage HTML content. |
| 27  | PctExtHyperlinks             | Continuous | Counts the percentage of external hyperlinks in webpage HTML source code.                                                                 |
| 28  | **PctExtResourceUrls**           | Continuous | Counts the percentage of external resource URLs in webpage HTML source code.                                                              |
| 29  | ExtFavicon                   | Binary      | Checks if the favicon is loaded from a domain name that is different from the webpage URL domain name.                                                |
| 30  | **InsecureForms**                | Binary      | Checks if the form action attribute contains a URL without HTTPS protocol.                                                                            |
| 31  | RelativeFormAction           | Binary      | Checks if the form action attribute contains a relative URL.                                                                                          |
| 32  | **ExtFormAction**                | Binary      | Checks if the form action attribute contains a URL from an external domain.                                                                     |
| 33  | AbnormalFormAction           | Categorical | Check if the form action attribute contains a “#”, “about:blank”, an empty string, or “javascript:true”.                                              |
| 34  | PctNullSelfRedirectHyperlinks| Continuous | Counts the percentage of hyperlinks fields containing empty value, self-redirect value such as “#”, the URL of current webpage, or some abnormal value such as “file://E:/”. |
| 35  | FrequentDomainNameMismatch   | Binary      | Checks if the most frequent domain name in HTML source code does not match the webpage URL domain name.                                                     |
| 36  | FakeLinkInStatusBar          | Binary      | Checks if HTML source code contains JavaScript command onMouseOver to display a fake URL in the status bar.                                     |
| 37  | RightClickDisabled           | Binary      | Checks if HTML source code contains JavaScript command to disable right click function.                                                          |
| 38  | **PopUpWindow**                  | Binary      | Checks if HTML source code contains JavaScript command to launch pop-ups.                                                                   |
| 39  | SubmitInfoToEmail            | Binary      | Check if HTML source code contains the HTML “mailto” function.                                                                                        |
| 40  | **IframeOrFrame**                | Binary      | Checks if iframe or frame is used in HTML source code.                                                                                                |
| 41  | MissingTitle                 | Binary      | Checks if the title tag is empty in HTML source code.                                                                                                  |
| 42  | **ImagesOnlyInForm**             | Binary      | Checks if the form scope in HTML source code contains no text at all but images only.                                                                      |
| 43  | SubdomainLevelRT             | Categorical | Counts the number of dots in hostname part of webpage URL. Apply rules and thresholds to generate value.                                              |
| 44  | **UrlLengthRT**                  | Categorical | Counts the total characters in the webpage URL. Apply rules and thresholds to generate value.                                                    |
| 45  | PctExtResourceUrlsRT         | Categorical | Counts the percentage of external resource URLs in webpage HTML source code. Apply rules and thresholds to generate value.                             |
| 46  | **AbnormalExtFormActionR**       | Categorical | Check if the form action attribute contains a foreign domain, “about:blank” or an empty string. Apply rules to generate value.                         |
| 47  | ExtMetaScriptLinkRT          | Categorical | Counts percentage of meta, script and link tags containing external URL in the attributes. Apply rules and thresholds to generate value.               |
| 48  | **PctExtNullSelfRedirectHyperlinksRT**| Categorical | Counts the percentage of hyperlinks in HTML source code that uses different domain names, starts with “#”, or using “JavaScript ::void(0)”. Apply rules and thresholds to generate value. |

# Feature Explaination
## SubdomainLevel
Level of subdomains in URL.
E.g.: https://**subdomain1**.**subdomain2**.domain.de -> 2

## UrlLength
Total characters of in URL.
E.g.: https://subdomain1.subdomain2.domain.de -> 39

## NumDashInHostname
Number of dashes (-) in hostname.
The hostname is everything between the protocol and the first / appearing after the top-level domain.
E.g.: https://**subdomain-1.subdomain-2.domain-0.de**/path-to-image ->  3

## TildeSymbol
If a tile symbol exists in URL.
E.g.: https://subdomain1.subdomain2.**~**domain.de
-> true

## NumPercent
Number of percent symbols (%) in URL.
E.g.: https://subdomain1.subdomain2%.domain%.de -> 2


## NumAmpersand
Number of ampersands (&) in URL.
E.g.: https://subdomain1.subdomain2.domain.de?var1=0**&**var2=1**&**var3=2 -> 2

## NumNumericChars
Number of numbers in URL.
E.g.: https://subdomain**1**.subdomain**2**.domain.de -> 2

## RandomString
If a random string is in URL.
A random string is ???
-> No source available (paper access restricted)

## DomainInSubdomains
If top-level-domain or country-code top-level-domain is part of a subdomain in URL.
See: https://en.wikipedia.org/wiki/List_of_Internet_top-level_domains, https://en.wikipedia.org/wiki/Country_code_top-level_domain#Lists

E.g.: https://subdomain1de.subdomain2name.domain.de -> true

## HttpsInHostname
If https in in hostname(?).
E.g.: https://https-subdomain1.subdomain2.domain.de -> true

## PathLength
Number of characters in paht in URL.
The path is between the hostname and a file: \<protocol>://\<host>/\<path>/\<file>?\<query>
E.g.: https://subdomain1.subdomain2.domain.de/path-to-subfolder/path-to-file/file.fl -> 30

## DoubleSlashInPath
If double slashes are in path in URL.
E.g.: https://subdomain1.subdomain2.domain.de/path//to/somwhere -> true

## EmbeddedBrandName
If brand name is in subdomains and paths in URL.
The brand name is the most frequent domain keyword in the HTML links of the html content.
E.g.: https://**subdomain1**.**subdomain2-brandname**.domain.de/**brandname/path/to/somwhere** -> true (2)

## PctExtResourceUrls
Percentage of external resource URLs of all URLS in html content.
An external resource URL is an URL that does point to a foreign domain.
E.g.: https://subdomain1.subdomain2.domain.de
- href="https://paypal.com/" -> external
- href="https://google.de/" -> external
- href="https://domain.de/path" -> internal
-> 2/3

## InsecureForms
If a form action attribute contains URL without https.
E.g.: \<form action="http://domain.de">...\</form> -> true

## ExtFormAction
If a form action attribute contains external URL.
E.g.: \<form action="https://evil.de">...\</form> -> true

## PctNullSelfRedirectHyperlinks
Percentage of hyperlinks containing empty values, self-redirect values, URL of current webpage or abnormal value(???) in html content.
Empty value: ""
Self-redirect values: "#", javascript::void(0)
Abnormal value: "file://E:/"

## FakeLinkInStatusBar
If html content contains JavScript command onMouseOver to display fake URL in status bar.
The status bar is displayed on the left/right lower corner of the browser to indicate the destination of a link or resource.
If onmouseover change the status bar -> Consider it a fake manipulation.

## PopUpWindow
If html content contains JavaScript command to launch pop-ups.
Command to launch pop-ups: window.open("...")

## IframeOrFrame
If frame or iframe are used in html content.
\<iframe ...>\</iframe>
\<frame>...

## ImagesOnlyInForm
If forms in html content does not contain any text and instead only images.
\<form ...>\<img/>\</form>

## UrlLengthRT
Number of characters in the whole URL.
E.g.: https://subdomain1.subdomain2.domain.de/path-to-subfolder/path-to-file/file.fl -> 78 

## AbnormalExtFormActionR
If form action contains foreign domain, "about:blank" or empty string.

## PctExtNullSelfRedirectHyperlinksRT
Percentage of hyperlinks in html content that uses differen domain names, start with ' or usess JavaScript ::void(0).

## 1. NumDots
Number of dots overall

## 3. PathLevel
Path depths = how many slashes are after the domain

## 5. NumDash
Number of - in URL
	
## 7. AtSymbol
@ symbol in the link? true or false
	
## 9. NumUnderscore
Number of _ in URL
	
## 11. NumQueryComponents
?
	
## 13. NumHash
Number of # in URL
	
## 15. NoHttps
https? true or false
	
## 17. IpAddress
Is hostname a domain or ip address?
extractable with checking if 4 numbers in hostname or not
	
## 19. DomainInPaths
Is top level domain in the path (.com, .de, etc)? true or false
--> does it occur after the hostnames '/'
	
## 21. HostnameLength
How many characters in hostname (www.google.com has 14)

## 23. QueryLength
How many characters after '?'
	
## 25. NumSensitiveWords
How many of these words are in the URL:  “secure”, “account”, “webscr”, “login”, “ebayisapi”, “signin”, “banking”, “confirm"
	
## 27. PctExtHyperlinks
?
	
## 29. ExtFavicon
?
	
## 31. RelativeFormAction
?
	
## 33. AbnormalFormAction
to do
	
## 35. FrequentDomainNameMismatch
?
	
## 37. RightClickDisabled
?
	
## 39. SubmitInfoToEmail
is in html source code the 'mailto' function? true/false

## 41. MissingTitle
is title tag in html code empty? true/false
	
## 43. SubdomainLevelRT
?
	
## 45. PctExtResourceUrlsRT
?
	
## 47. ExtMetaScriptLinkRT
?

