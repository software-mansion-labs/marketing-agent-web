from pydantic import BaseModel, Field


class WebsiteHeader(BaseModel):
    """Website URL for identification."""

    link: str = Field(description="URL of the website")


class Website(BaseModel):
    """Website header and content."""

    header: WebsiteHeader = Field(description="website identification")
    content: str = Field(description="website contents")


class WebsiteChoice(BaseModel):
    """Website and justification why it's a good pick."""

    website: WebsiteHeader = Field(description="website")
    justification: str = Field(description="why it's a good place for our purposes")


class WebsiteChoiceList(BaseModel):
    """List of Website choices."""

    websites: list[WebsiteChoice] = Field(description="list of picked websites")


class Critique(BaseModel):
    """Content suitability critique."""

    upsides: str = Field(description="suitability upsides")
    downsides: str = Field(description="suitability downsides")


class WebsiteCritique(BaseModel):
    """Website suitability critique."""

    website: WebsiteHeader = Field(description="website")
    critique: Critique = Field(description="critique of its suitability")
