# A page for every course in _data/courses.yml with `show: true`, at /teaching/<slug>/, laid out by
# _layouts/course.liquid. Adding a course to the data file is enough to publish its page.
#
# Each shown course also gets, for the templates:
# - `url`, the address of its page, which the lessons, the blog's rows and the markup link to;
# - `number`, its place on the route on /teaching/.
# The page takes its title and search snippet from the course, its picture for link previews from the course's card
# (bin/course-card.py draws them), and the date of the newest lesson as its last change (for sitemap.xml); /teaching/
# gets the date of the newest lesson of all.
module CoursePages
  class Generator < Jekyll::Generator
    safe true
    priority :high # before the generators that list pages

    def generate(site)
      posts = site.posts.docs.to_h { |post| [post.url, post] }
      shown = (site.data["courses"] || []).select { |course| course["show"] }
      shown.each_with_index do |course, k|
        slug = course["slug"] or raise "course #{course['id']}: no slug in _data/courses.yml"
        course["url"] = "/teaching/#{slug}/"
        course["number"] = k + 1
        lessons = (course["lessons"] || []).select { |lesson| lesson["url"] }
        missing = lessons.map { |lesson| lesson["url"] }.reject { |url| posts.key?(url) }
        raise "course #{course['id']}: no post at #{missing.join(', ')}" unless missing.empty?
        newest = lessons.map { |lesson| posts[lesson["url"]] }.max_by(&:date)

        page = Jekyll::PageWithoutAFile.new(site, site.source, "teaching/#{slug}", "index.html")
        page.data.merge!(
          "layout" => "course",
          "course_id" => course["id"],
          "title" => course["title"],
          "description" => course["summary"],
          "seo_title" => course["seo_title"],
          "seo_description" => course["seo_description"]
        )
        # the course's own card (bin/course-card.py), or else the newest lesson's cover
        card = "assets/img/teaching/card-#{slug}.png"
        if File.exist?(File.join(site.source, card))
          page.data["og_image"] = "#{site.config['url']}/#{card}"
          page.data["og_image_alt"] = "#{course['title']}, a course by #{site.config['first_name']} #{site.config['last_name']}"
        elsif newest&.data&.fetch("og_image", nil)
          page.data["og_image"] = newest.data["og_image"]
        end
        page.data["last_modified_at"] = newest.data["last_updated"] || newest.date if newest
        site.pages << page
      end
      dates = shown.flat_map { |course| course["lessons"] || [] }.filter_map { |lesson| posts[lesson["url"]]&.date }
      catalog = site.pages.find { |page| page.url == "/teaching/" }
      catalog.data["last_modified_at"] = dates.max if catalog && !dates.empty?
    end
  end
end
