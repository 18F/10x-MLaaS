![MlaaS](https://i.postimg.cc/vZDzYQb2/mlaas2.jpg)
# 10x Machine Learning as a Service
---
[Formerly known as Qualitative Data Management]

We are currently in the **third phase of [10x](tenx): Development**. We estimate this phase to last approximately 6 months and will end in May 2019.

## Overview

> USA.gov and Gobierno.USA.gov provide a vital service: connecting members of the public with official information about government services. In an effort to better serve the public, both sites collect (and act on) feedback from users. They collect this feedback through various channels: call centers, the chat feature, sitewide surveys, and page-level surveys. (Our research focused almost entirely on page-level surveys.) For context, page-level surveys appear on “nearly every content page, directory record page, and state details page” — in other words, pages with the content that users need to answer their questions.

### The problem statement

> As a government employee, how can I more quickly and effectively analyze comments provided by site visitors to identify timely issues, improve the usability of the site and information provided, and further direct visitors to important services and information they need?

### Our challenge

> Help the USAgov team better serve their users by (a) introducing process improvements and simple automation in analyzing open-ended comments submitted through their website and (b) introduce experimental sentiment-analysis tools and other qualitative data analysis techniques to more robustly analyze these data with less human intervention.

As expected, the scope of our project has shifted to offering these machine learning tools to the entire Office of Customer Experience. During Phase II, we prototyped and delivered a machine learning tool to aid the USAgov team but we believe this tool (or similar SaaS) could be leveraged to reduce the burden on other teams in the Office of Customer Experience, as well as outside GSA.

## Who are we?

Team members:

- Amos Stone, Innovation Specialist, [18F](eighteenf)
- Colin Craig, Innovation Specialist, [18F](eighteenf)
- Will Cahoe, Program Analyst, [10x](tenx)


Advisers:

- David Kaufmann, USAgov Analytics, [Office of Products and Platforms](https://www.gsa.gov/about-us/organization/federal-acquisition-service/technology-transformation-services/office-of-products-and-programs)
- Marybeth Murphy, USAgov Analytics, [Office of Products and Platforms](https://www.gsa.gov/about-us/organization/federal-acquisition-service/technology-transformation-services/office-of-products-and-programs)

Former team members:
- Chris Goranson, research, [18F](tenx)
- Kate Saul, research, [18F](eighteenf)


## Progress

We are tracking the work for this Phase on our [Kanban board](https://github.com/18F/10x-MLaaS/projects/2).

Any issues or ideas that we want to keep track of for later are being noted in
the [GitHub issues](https://github.com/18F/10x-MLaaS/issues).

We post weekly progress updates in [updates](updates).


## Investigation

1. **USA.gov’s data-management process is entirely manual.** Although the team uses HHS’s Voice of the Customer tool to capture survey data, all of the review and analysis are manual. 
1. **This manual process takes time and creates significant challenges.** Manual review and analysis take a considerable amount of time — time that could be spent creating more effective content and replying to urgent user inquiries.    
1. **Workflow improvements would complement automation.** In addition to automating data processing, identifying a product owner, documenting the workflow, and finding other ways to streamline the process would increase efficiency. 
1. **Automation improvements will speed up the workflow improvements and reduce errors.**  Steps that are largely duplicative between analysis periods can be automated, thereby allowing the USA.gov team to spend more time gleaning insights from the valuable data. 
1. **The USA.gov team enthusiastically seeks process improvements.** Everyone we spoke to emphasized the need for process improvements and an openness toward change.  
1. **Novel approaches to qualitative data enhancements can be applied without getting in the way.**  Once workflow and automation enhancements are complete, the analyses can be further explored using methods and tools that work well for analyzing qualitative data (natural language processing, sentiment analysis).  
1. **We recommend moving forward with this project.** Based on the potential for improving USA.gov’s service offerings, the applicability of automation tactics to other federal agencies and other qualitative data held by the government, the impact improvements will have on the lives of American citizens, and the team’s openness to change, we recommend that the next phase of this project should be funded. 

The full Phase I investigation report is available [here](https://docs.google.com/document/d/1InUpl7v3wa0v05JYCB8-atoDene9-Gzbz-ELY7OPVKY/).


## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for additional information.

Join us in
[#10x-mlaas](https://gsa-tts.slack.com/messages/C9QNC7STG) or [ask
us a question](https://github.com/18F/10x-MLaaSw/issues/new).


## Public domain

This project is in the worldwide [public domain](LICENSE.md). As stated in [CONTRIBUTING](CONTRIBUTING.md):

> This project is in the public domain within the United States, and copyright and related rights in the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/).
>
> All contributions to this project will be released under the CC0 dedication. By submitting a pull request, you are agreeing to comply with this waiver of copyright interest.

[tts]: https://www.gsa.gov/about-us/organization/federal-acquisition-service/technology-transformation-services
[tenx]: https://10x.gsa.gov
[eighteenf]: https://18f.gsa.gov
